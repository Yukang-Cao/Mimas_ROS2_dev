// xmaxx_protocol.h
#pragma once
#include <cstdint>
#include <cstddef>

constexpr uint32_t XMAXX_MAGIC   = 0x5A5A5A5A;
constexpr uint8_t  XMAXX_VERSION = 0x01;
constexpr uint8_t  TYPE_TELEM    = 0x81;
constexpr uint8_t  TYPE_DRIVE    = 0x01;

constexpr std::size_t TELEMETRY_MSG_SIZE = 37;

#pragma pack(push, 1)
struct MsgHeader {
    uint32_t magic;   // 'ZZZZ' little-endian
    uint8_t  version; // 0x01
    uint8_t  type;    // e.g., 0x81 telemetry
    uint8_t  crc;     // CRC-8 over (header with crc=0) + body
};

struct Telemetry {
    uint64_t counter;       // Q
    uint8_t  state;         // B
    uint16_t rcThrottle;    // H
    uint16_t rcSteering;    // H
    uint16_t rcSwitchA;     // H
    uint16_t rcSwitchB;     // H
    uint16_t rcSwitchC;     // H
    uint16_t rcSwitchD;     // H
    uint16_t acThrottle;    // H
    uint16_t acSteering;    // H
    uint8_t  upRssi;        // B
    uint8_t  upLqi;         // B
    uint8_t  downRssi;      // B
    uint8_t  downLqi;       // B
    uint16_t escVoltageRaw; // H
    uint16_t escCurrentRaw; // H
    uint16_t escRpmRaw;     // H
    uint16_t escTempRaw;    // H
};
#pragma pack(pop)

static_assert(sizeof(Telemetry) == TELEMETRY_MSG_SIZE, "Telemetry must be 37 bytes packed");
