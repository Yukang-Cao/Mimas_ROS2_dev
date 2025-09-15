// xmaxx.cpp
// g++ -std=c++17 -O2 -pthread xmaxx.cpp -o xmaxx

// 1. Convert this to a ROS2 script

// 2. /home/titan/ros2_ws/src/radio_telem_to_cmdvel/radio_telem_to_cmdvel/send_cmd_node.py : 
// this is the reference python file based on which this file is based on.

// 3. add a function send_rc_cmd() to send commands. So setup a serial.

// #include "xmaxx.h"
#include <xmaxx_bringup/xmaxx.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <vector>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

// Expect these in your project:
extern "C" {
    // none
}
#include <cstdint>
#include <cstdio>

Xmaxx::Xmaxx(std::string dev, int baud)
    : dev_(std::move(dev)), baud_(baud) {}

Xmaxx::~Xmaxx() { stop(); }

bool Xmaxx::start() {
    if (running_.load()) return true;
    if (fd_ < 0) {
        fd_ = openSerial_();
        if (fd_ < 0) return false;
    }
    running_.store(true);
    thread_ = std::thread(&Xmaxx::readerLoop_, this);
    return true;
}

void Xmaxx::stop() {
    if (!running_.exchange(false)) {
        // not running
    }
    // Closing the fd will unblock any blocking read
    closeSerial_();
    if (thread_.joinable()) thread_.join();
}

bool Xmaxx::sendDriveCmd(uint16_t throttle, uint16_t steering) {
    if (fd_ < 0 || !running_.load()) {
        return false;
    }

    // Drive message structure: magic(4) + version(1) + type(1) + crc(1) + throttle(2) + steering(2) = 11 bytes
    struct DriveMsg {
        uint32_t magic;
        uint8_t version;
        uint8_t type;
        uint8_t crc;
        uint16_t throttle;
        uint16_t steering;
    } __attribute__((packed));

    DriveMsg msg{};
    msg.magic = 0x5A5A5A5A;
    msg.version = 0x01;
    msg.type = 0x01;  // TYPE_DRIVE
    msg.crc = 0;      // Calculate after
    msg.throttle = throttle;
    msg.steering = steering;

    // Calculate CRC over header (with crc=0) + body (throttle + steering)
    uint8_t crc = crc8_(reinterpret_cast<const uint8_t*>(&msg), sizeof(msg));
    msg.crc = crc;

    // Write the complete message
    ssize_t written = ::write(fd_, &msg, sizeof(msg));
    if (written != sizeof(msg)) {
        perror("write drive command");
        return false;
    }
    return true;
}

// --------- private helpers ---------

uint8_t Xmaxx::crc8_(const uint8_t* data, size_t len) {
    uint8_t c = 0;
    for (size_t i = 0; i < len; ++i) {
        c ^= data[i];
        for (int j = 0; j < 8; ++j) {
            if (c & 0x80) c = static_cast<uint8_t>((c << 1) ^ 0xE7);
            else          c = static_cast<uint8_t>(c << 1);
        }
    }
    return c;
}

bool Xmaxx::setInterfaceAttribs_(int fd, int speed) {
    termios tty{};
    if (tcgetattr(fd, &tty) != 0) {
        perror("tcgetattr");
        return false;
    }

    cfmakeraw(&tty);

    speed_t brate;
    switch (speed) {
        case 460800: brate = B460800; break;
        case 230400: brate = B230400; break;
        case 115200: brate = B115200; break;
        default:     brate = B460800; break;
    }
    cfsetispeed(&tty, brate);
    cfsetospeed(&tty, brate);

    // 8N1
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;

    // No flow control
    tty.c_cflag &= ~CRTSCTS;
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);

    // Blocking read: at least 1 byte, no inter-byte timeout
    tty.c_cc[VMIN]  = 1;
    tty.c_cc[VTIME] = 0;

    // Enable receiver, ignore modem ctrl lines
    tty.c_cflag |= (CLOCAL | CREAD);

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        perror("tcsetattr");
        return false;
    }
    return true;
}

int Xmaxx::openSerial_() {
    int fd = ::open(dev_.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
    if (fd < 0) {
        perror("open");
        return -1;
    }
    if (!setInterfaceAttribs_(fd, baud_)) {
        ::close(fd);
        return -1;
    }
    return fd;
}

void Xmaxx::closeSerial_() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

bool Xmaxx::readExact_(uint8_t* buf, size_t len) {
    size_t got = 0;
    while (running_.load() && got < len) {
        ssize_t n = ::read(fd_, buf + got, len - got);
        if (n > 0) {
            got += static_cast<size_t>(n);
        } else if (n == 0) {
            return false; // EOF unexpected on ttys
        } else {
            if (errno == EINTR) continue;
            // If fd_ was closed from another thread, bail out quietly
            if (errno == EBADF || errno == EIO) return false;
            perror("read");
            return false;
        }
    }
    return got == len;
}

// --------- reader thread ---------

void Xmaxx::readerLoop_() {
    bool sync = false;
    std::string msg_accum;
    std::vector<uint8_t> headerBuf(sizeof(MsgHeader));

    try {
        while (running_.load()) {
            // --- Acquire header (sync on 'ZZZZ') ---
            if (!sync) {
                msg_accum.clear();
                size_t zcount = 0;
                for (;;) {
                    if (!running_.load()) return;
                    uint8_t b;
                    if (!readExact_(&b, 1)) { std::cerr << "[!] read error\n"; return; }
                    if (b == 'Z') {
                        if (zcount < headerBuf.size()) headerBuf[zcount] = b;
                        ++zcount;
                        if (zcount == 4) {
                            // Read rest of header (3 bytes)
                            if (!readExact_(headerBuf.data() + 4, headerBuf.size() - 4)) {
                                std::cerr << "[!] read error (header tail)\n"; return;
                            }
                            std::cout << "[-] MSG: " << msg_accum << "\n";
                            std::cout << "[-] Got sync...\n";
                            sync = true;
                            break;
                        }
                    } else {
                        if (b >= 32 && b < 127) msg_accum.push_back(static_cast<char>(b));
                        zcount = 0;
                    }
                }
            } else {
                if (!readExact_(headerBuf.data(), headerBuf.size())) {
                    std::cerr << "[!] read error (header)\n"; return;
                }
            }

            // --- Parse header ---
            MsgHeader hdr{};
            std::memcpy(&hdr, headerBuf.data(), sizeof(hdr));

            if (hdr.magic != 0x5A5A5A5A) {
                std::cerr << "[!] Bad magic\n";
                sync = false;
                continue;
            }
            if (hdr.version != 0x01) {
                std::cerr << "[!] Bad version\n";
                sync = false;
                continue;
            }

            // --- Read body (largest union = Telemetry = 37 bytes) ---
            std::array<uint8_t, sizeof(Telemetry)> body{};
            if (!readExact_(body.data(), body.size())) {
                std::cerr << "[!] read error (body)\n"; return;
            }

            // --- CRC over header with crc=0 + body ---
            MsgHeader hdr_zero = hdr;
            hdr_zero.crc = 0;
            std::array<uint8_t, sizeof(MsgHeader) + sizeof(Telemetry)> crcbuf{};
            std::memcpy(crcbuf.data(), &hdr_zero, sizeof(hdr_zero));
            std::memcpy(crcbuf.data() + sizeof(hdr_zero), body.data(), body.size());
            uint8_t computed = crc8_(crcbuf.data(), crcbuf.size());

            if (hdr.crc != computed) {
                std::cerr << "[!] Bad CRC (got 0x" << std::hex << int(hdr.crc)
                          << ", want 0x" << int(computed) << std::dec << ")\n";
                sync = false;
                continue;
            }

            // --- Dispatch ---
            if (hdr.type == 0x81) {
                Telemetry t{};
                std::memcpy(&t, body.data(), sizeof(Telemetry));
                if (on_telem_) {
                    on_telem_(t);
                } else {
                    // Fallback print (matches your original code)
                    std::cout << "[-] telem : ("
                              << t.counter << ", "
                              << int(t.state) << ", "
                              << t.rcThrottle << ", "
                              << t.rcSteering << ", "
                              << t.rcSwitchA << ", "
                              << t.rcSwitchB << ", "
                              << t.rcSwitchC << ", "
                              << t.rcSwitchD << ", "
                              << t.acThrottle << ", "
                              << t.acSteering << ", "
                              << int(t.upRssi) << ", "
                              << int(t.upLqi) << ", "
                              << int(t.downRssi) << ", "
                              << int(t.downLqi) << ", "
                              << t.escVoltageRaw << ", "
                              << t.escCurrentRaw << ", "
                              << t.escRpmRaw << ", "
                              << t.escTempRaw
                              << ")\n";
                }
            } else {
                std::cout << "[-] msg type 0x" << std::hex << int(hdr.type)
                          << std::dec << " (ignored)\n";
            }
        }
    } catch (...) {
        std::cerr << "[!] Exception in readerLoop_\n";
    }
}


/* // Main is commented out because this file will be used as a library inside xmaxx_bringup package
int main() {
    Xmaxx iface;
    iface.setTelemetryCallback([](const Telemetry& t){
        // handle telemetry
    });
    if (!iface.start()) return 1;

    // run forever
    for (;;) std::this_thread::sleep_for(std::chrono::seconds(1));
}
*/