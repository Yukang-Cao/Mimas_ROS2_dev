// xmaxx.h
#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>
// #include "xmaxx_protocol.h"
#include <xmaxx_bringup/xmaxx_protocol.h>
// #include <xmaxx_bringup/xmaxx_telem.h>


struct MsgHeader;   // from xmaxx_protocol.h
struct Telemetry;   // from xmaxx_protocol.h

constexpr const char* kDefaultSerialDev = "/dev/ttyTHS1";
constexpr int         kDefaultBaud      = 460800;

class Xmaxx
{
public:
    using TelemetryCallback = std::function<void(const Telemetry&)>;

    explicit Xmaxx(std::string dev = kDefaultSerialDev, int baud = kDefaultBaud);
    ~Xmaxx();

    Xmaxx(const Xmaxx&) = delete;
    Xmaxx& operator=(const Xmaxx&) = delete;
    Xmaxx(Xmaxx&&) noexcept = delete;
    Xmaxx& operator=(Xmaxx&&) noexcept = delete;

    bool start();              // open + spawn reader
    void stop();               // stop reader + close

    bool isRunning() const { return running_.load(); }
    bool isOpen()    const { return fd_ >= 0; }
    int  fd()        const { return fd_; }

    void setTelemetryCallback(TelemetryCallback cb) { on_telem_ = std::move(cb); }

    // Optional: implement if you have a TX message format
    bool sendDriveCmd(uint16_t /*throttle*/, uint16_t /*steering*/);

private:
    // --- serial helpers ---
    bool setInterfaceAttribs_(int fd, int speed);
    int  openSerial_();
    void closeSerial_();
    bool readExact_(uint8_t* buf, size_t len);
    static uint8_t crc8_(const uint8_t* data, size_t len);

    // --- thread loop ---
    void readerLoop_();

private:
    std::string dev_;
    int         baud_;
    int         fd_{-1};

    std::thread         thread_;
    std::atomic<bool>   running_{false};
    TelemetryCallback   on_telem_{};
};
