//===-- unittests/Runtime/ExternalIOTest.cpp --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Sanity test for all external I/O modes
//
//===----------------------------------------------------------------------===//

#include "CrashHandlerFixture.h"
#include "gtest/gtest.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang/Runtime/io-api.h"
#include "flang/Runtime/main.h"
#include "flang/Runtime/stop.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
#include <string_view>

using namespace Fortran::runtime;
using namespace Fortran::runtime::io;

struct ExternalIOTests : public CrashHandlerFixture {};

TEST(ExternalIOTests, TestDirectUnformatted) {
  // OPEN(NEWUNIT=unit,ACCESS='DIRECT',ACTION='READWRITE',&
  //   FORM='UNFORMATTED',RECL=8,STATUS='SCRATCH')
  Cookie io{IONAME(BeginOpenNewUnit)(__FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(SetAccess)(io, "DIRECT", 6)) << "SetAccess(DIRECT)";
  ASSERT_TRUE(IONAME(SetAction)(io, "READWRITE", 9)) << "SetAction(READWRITE)";
  ASSERT_TRUE(IONAME(SetForm)(io, "UNFORMATTED", 11)) << "SetForm(UNFORMATTED)";

  std::int64_t buffer;
  static constexpr std::size_t recl{sizeof buffer};
  ASSERT_TRUE(IONAME(SetRecl)(io, recl)) << "SetRecl()";
  ASSERT_TRUE(IONAME(SetStatus)(io, "SCRATCH", 7)) << "SetStatus(SCRATCH)";

  int unit{-1};
  ASSERT_TRUE(IONAME(GetNewUnit)(io, unit)) << "GetNewUnit()";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for OpenNewUnit";

  StaticDescriptor<0> staticDescriptor;
  Descriptor &desc{staticDescriptor.descriptor()};
  desc.Establish(TypeCode{CFI_type_int8_t}, recl, &buffer, 0);
  desc.Check();

  // INQUIRE(IOLENGTH=) j
  io = IONAME(BeginInquireIoLength)(__FILE__, __LINE__);
  ASSERT_TRUE(IONAME(OutputDescriptor)(io, desc))
      << "OutputDescriptor() for InquireIoLength";
  ASSERT_EQ(IONAME(GetIoLength)(io), recl) << "GetIoLength";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for InquireIoLength";

  static constexpr int records{10};
  for (int j{1}; j <= records; ++j) {
    // WRITE(UNIT=unit,REC=j) j
    io = IONAME(BeginUnformattedOutput)(unit, __FILE__, __LINE__);
    ASSERT_TRUE(IONAME(SetRec)(io, j)) << "SetRec(" << j << ')';

    buffer = j;
    ASSERT_TRUE(IONAME(OutputDescriptor)(io, desc))
        << "OutputDescriptor() for Write";

    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for Write";
  }

  for (int j{records}; j >= 1; --j) {
    buffer = -1;
    // READ(UNIT=unit,REC=j) n
    io = IONAME(BeginUnformattedInput)(unit, __FILE__, __LINE__);
    ASSERT_TRUE(IONAME(SetRec)(io, j)) << "SetRec(" << j << ')';
    ASSERT_TRUE(IONAME(InputDescriptor)(io, desc))
        << "InputDescriptor() for Read";

    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for Read";

    ASSERT_EQ(buffer, j) << "Read back " << buffer
                         << " from direct unformatted record " << j
                         << ", expected " << j << '\n';
  }
  // CLOSE(UNIT=unit,STATUS='DELETE')
  io = IONAME(BeginClose)(unit, __FILE__, __LINE__);
  ASSERT_TRUE(IONAME(SetStatus)(io, "DELETE", 6)) << "SetStatus(DELETE)";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for Close";
}

TEST(ExternalIOTests, TestDirectUnformattedSwapped) {
  // OPEN(NEWUNIT=unit,ACCESS='DIRECT',ACTION='READWRITE',&
  //   FORM='UNFORMATTED',RECL=8,STATUS='SCRATCH',CONVERT='NATIVE')
  auto *io{IONAME(BeginOpenNewUnit)(__FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(SetAccess)(io, "DIRECT", 6)) << "SetAccess(DIRECT)";
  ASSERT_TRUE(IONAME(SetAction)(io, "READWRITE", 9)) << "SetAction(READWRITE)";
  ASSERT_TRUE(IONAME(SetForm)(io, "UNFORMATTED", 11)) << "SetForm(UNFORMATTED)";
  ASSERT_TRUE(IONAME(SetConvert)(io, "NATIVE", 6)) << "SetConvert(NATIVE)";

  std::int64_t buffer;
  static constexpr std::size_t recl{sizeof buffer};
  ASSERT_TRUE(IONAME(SetRecl)(io, recl)) << "SetRecl()";
  ASSERT_TRUE(IONAME(SetStatus)(io, "SCRATCH", 7)) << "SetStatus(SCRATCH)";

  int unit{-1};
  ASSERT_TRUE(IONAME(GetNewUnit)(io, unit)) << "GetNewUnit()";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for OpenNewUnit";

  StaticDescriptor<0> staticDescriptor;
  Descriptor &desc{staticDescriptor.descriptor()};
  desc.Establish(TypeCode{CFI_type_int64_t}, recl, &buffer, 0);
  desc.Check();

  static constexpr int records{10};
  for (int j{1}; j <= records; ++j) {
    // WRITE(UNIT=unit,REC=j) j
    io = IONAME(BeginUnformattedOutput)(unit, __FILE__, __LINE__);
    ASSERT_TRUE(IONAME(SetRec)(io, j)) << "SetRec(" << j << ')';
    buffer = j;
    ASSERT_TRUE(IONAME(OutputDescriptor)(io, desc))
        << "OutputDescriptor() for Write";
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for Write";
  }

  // OPEN(UNIT=unit,STATUS='OLD',CONVERT='SWAP')
  io = IONAME(BeginOpenUnit)(unit, __FILE__, __LINE__);
  ASSERT_TRUE(IONAME(SetStatus)(io, "OLD", 3)) << "SetStatus(OLD)";
  ASSERT_TRUE(IONAME(SetConvert)(io, "SWAP", 4)) << "SetConvert(SWAP)";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for OpenUnit";

  for (int j{records}; j >= 1; --j) {
    // READ(UNIT=unit,REC=j) n
    io = IONAME(BeginUnformattedInput)(unit, __FILE__, __LINE__);
    ASSERT_TRUE(IONAME(SetRec)(io, j)) << "SetRec(" << j << ')';
    ASSERT_TRUE(IONAME(InputDescriptor)(io, desc))
        << "InputDescriptor() for Read";
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for Read";
    ASSERT_EQ(buffer >> 56, j)
        << "Read back " << (buffer >> 56) << " from direct unformatted record "
        << j << ", expected " << j << '\n';
  }

  // CLOSE(UNIT=unit,STATUS='DELETE')
  io = IONAME(BeginClose)(unit, __FILE__, __LINE__);
  ASSERT_TRUE(IONAME(SetStatus)(io, "DELETE", 6)) << "SetStatus(DELETE)";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for Close";
}

TEST(ExternalIOTests, TestSequentialFixedUnformatted) {
  // OPEN(NEWUNIT=unit,ACCESS='SEQUENTIAL',ACTION='READWRITE',&
  //   FORM='UNFORMATTED',RECL=8,STATUS='SCRATCH')
  auto *io{IONAME(BeginOpenNewUnit)(__FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(SetAccess)(io, "SEQUENTIAL", 10))
      << "SetAccess(SEQUENTIAL)";
  ASSERT_TRUE(IONAME(SetAction)(io, "READWRITE", 9)) << "SetAction(READWRITE)";
  ASSERT_TRUE(IONAME(SetForm)(io, "UNFORMATTED", 11)) << "SetForm(UNFORMATTED)";

  std::int64_t buffer;
  static constexpr std::size_t recl{sizeof buffer};

  ASSERT_TRUE(IONAME(SetRecl)(io, recl)) << "SetRecl()";
  ASSERT_TRUE(IONAME(SetStatus)(io, "SCRATCH", 7)) << "SetStatus(SCRATCH)";

  int unit{-1};
  ASSERT_TRUE(IONAME(GetNewUnit)(io, unit)) << "GetNewUnit()";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for OpenNewUnit";

  // INQUIRE(IOLENGTH=) j, ...
  StaticDescriptor<0> staticDescriptor;
  Descriptor &desc{staticDescriptor.descriptor()};
  desc.Establish(TypeCode{CFI_type_int64_t}, recl, &buffer, 0);
  desc.Dump(stderr);
  desc.Check();
  io = IONAME(BeginInquireIoLength)(__FILE__, __LINE__);
  for (int j{1}; j <= 3; ++j) {
    ASSERT_TRUE(IONAME(OutputDescriptor)(io, desc))
        << "OutputDescriptor() for InquireIoLength " << j;
  }
  ASSERT_EQ(IONAME(GetIoLength)(io), 3 * recl) << "GetIoLength";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for InquireIoLength";

  static const int records{10};
  for (int j{1}; j <= records; ++j) {
    // DO J=1,RECORDS; WRITE(UNIT=unit) j; END DO
    io = IONAME(BeginUnformattedOutput)(unit, __FILE__, __LINE__);
    buffer = j;
    ASSERT_TRUE(IONAME(OutputDescriptor)(io, desc))
        << "OutputDescriptor() for Write";
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for WRITE";
  }

  // REWIND(UNIT=unit)
  io = IONAME(BeginRewind)(unit, __FILE__, __LINE__);
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for Rewind";

  for (int j{1}; j <= records; ++j) {
    // DO J=1,RECORDS; READ(UNIT=unit) n; check n; END DO
    io = IONAME(BeginUnformattedInput)(unit, __FILE__, __LINE__);
    ASSERT_TRUE(IONAME(InputDescriptor)(io, desc))
        << "InputDescriptor() for Read";
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for Read";
    ASSERT_EQ(buffer, j) << "Read back " << buffer
                         << " from sequential fixed unformatted record " << j
                         << ", expected " << j << '\n';
  }

  for (int j{records}; j >= 1; --j) {
    // BACKSPACE(UNIT=unit)
    io = IONAME(BeginBackspace)(unit, __FILE__, __LINE__);
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for Backspace (before read)";
    // READ(UNIT=unit) n
    io = IONAME(BeginUnformattedInput)(unit, __FILE__, __LINE__);
    ASSERT_TRUE(IONAME(InputDescriptor)(io, desc))
        << "InputDescriptor() for Read";
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for Read";
    ASSERT_EQ(buffer, j) << "Read back " << buffer
                         << " from sequential fixed unformatted record " << j
                         << " after backspacing, expected " << j << '\n';
    // BACKSPACE(UNIT=unit)
    io = IONAME(BeginBackspace)(unit, __FILE__, __LINE__);
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for Backspace (after read)";
  }

  // CLOSE(UNIT=unit,STATUS='DELETE')
  io = IONAME(BeginClose)(unit, __FILE__, __LINE__);
  ASSERT_TRUE(IONAME(SetStatus)(io, "DELETE", 6)) << "SetStatus(DELETE)";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for Close";
}

TEST(ExternalIOTests, TestSequentialVariableUnformatted) {
  // OPEN(NEWUNIT=unit,ACCESS='SEQUENTIAL',ACTION='READWRITE',&
  //   FORM='UNFORMATTED',STATUS='SCRATCH')
  auto *io{IONAME(BeginOpenNewUnit)(__FILE__, __LINE__)};

  ASSERT_TRUE(IONAME(SetAccess)(io, "SEQUENTIAL", 10))
      << "SetAccess(SEQUENTIAL)";
  ASSERT_TRUE(IONAME(SetAction)(io, "READWRITE", 9)) << "SetAction(READWRITE)";
  ASSERT_TRUE(IONAME(SetForm)(io, "UNFORMATTED", 11)) << "SetForm(UNFORMATTED)";
  ASSERT_TRUE(IONAME(SetStatus)(io, "SCRATCH", 7)) << "SetStatus(SCRATCH)";

  int unit{-1};
  ASSERT_TRUE(IONAME(GetNewUnit)(io, unit)) << "GetNewUnit()";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for OpenNewUnit";

  static const int records{10};
  std::int64_t buffer[records]; // INTEGER*8 :: BUFFER(0:9) = [(j,j=0,9)]
  for (int j{0}; j < records; ++j) {
    buffer[j] = j;
  }

  StaticDescriptor<0> staticDescriptor;
  Descriptor &desc{staticDescriptor.descriptor()};

  for (int j{1}; j <= records; ++j) {
    // DO J=1,RECORDS; WRITE(UNIT=unit) BUFFER(0:j); END DO
    io = IONAME(BeginUnformattedOutput)(unit, __FILE__, __LINE__);
    desc.Establish(TypeCode{sizeof *buffer}, j * sizeof *buffer, buffer, 0);
    desc.Check();
    ASSERT_TRUE(IONAME(OutputDescriptor)(io, desc))
        << "OutputDescriptor() for Write";
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for Write";
  }

  // REWIND(UNIT=unit)
  io = IONAME(BeginRewind)(unit, __FILE__, __LINE__);
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for Rewind";
  for (int j{1}; j <= records; ++j) {
    // DO J=1,RECORDS; READ(UNIT=unit) n; check n; END DO
    io = IONAME(BeginUnformattedInput)(unit, __FILE__, __LINE__);
    desc.Establish(TypeCode{sizeof *buffer}, j * sizeof *buffer, buffer, 0);
    desc.Check();
    ASSERT_TRUE(IONAME(InputDescriptor)(io, desc))
        << "InputDescriptor() for Read";
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for Read";
    for (int k{0}; k < j; ++k) {
      ASSERT_EQ(buffer[k], k) << "Read back [" << k << "]=" << buffer[k]
                              << " from direct unformatted record " << j
                              << ", expected " << k << '\n';
    }
  }

  for (int j{records}; j >= 1; --j) {
    // BACKSPACE(unit)
    io = IONAME(BeginBackspace)(unit, __FILE__, __LINE__);
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for Backspace (before read)";
    // READ(unit=unit) n; check
    io = IONAME(BeginUnformattedInput)(unit, __FILE__, __LINE__);
    desc.Establish(TypeCode{sizeof *buffer}, j * sizeof *buffer, buffer, 0);
    desc.Check();
    ASSERT_TRUE(IONAME(InputDescriptor)(io, desc)) << "InputDescriptor()";
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for InputUnformattedBlock";
    for (int k{0}; k < j; ++k) {
      ASSERT_EQ(buffer[k], k) << "Read back [" << k << "]=" << buffer[k]
                              << " from sequential variable unformatted record "
                              << j << ", expected " << k << '\n';
    }
    // BACKSPACE(unit)
    io = IONAME(BeginBackspace)(unit, __FILE__, __LINE__);
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for Backspace (after read)";
  }

  // CLOSE(UNIT=unit,STATUS='DELETE')
  io = IONAME(BeginClose)(unit, __FILE__, __LINE__);
  ASSERT_TRUE(IONAME(SetStatus)(io, "DELETE", 6)) << "SetStatus(DELETE)";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for Close";
}

TEST(ExternalIOTests, TestDirectFormatted) {
  // OPEN(NEWUNIT=unit,ACCESS='DIRECT',ACTION='READWRITE',&
  //   FORM='FORMATTED',RECL=8,STATUS='SCRATCH')
  auto *io{IONAME(BeginOpenNewUnit)(__FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(SetAccess)(io, "DIRECT", 6)) << "SetAccess(DIRECT)";
  ASSERT_TRUE(IONAME(SetAction)(io, "READWRITE", 9)) << "SetAction(READWRITE)";
  ASSERT_TRUE(IONAME(SetForm)(io, "FORMATTED", 9)) << "SetForm(FORMATTED)";

  static constexpr std::size_t recl{8};
  ASSERT_TRUE(IONAME(SetRecl)(io, recl)) << "SetRecl()";
  ASSERT_TRUE(IONAME(SetStatus)(io, "SCRATCH", 7)) << "SetStatus(SCRATCH)";

  int unit{-1};
  ASSERT_TRUE(IONAME(GetNewUnit)(io, unit)) << "GetNewUnit()";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for OpenNewUnit";

  static constexpr int records{10};
  static const char fmt[]{"(I4)"};
  for (int j{1}; j <= records; ++j) {
    // WRITE(UNIT=unit,FMT=fmt,REC=j) j
    io = IONAME(BeginExternalFormattedOutput)(
        fmt, sizeof fmt - 1, nullptr, unit, __FILE__, __LINE__);
    ASSERT_TRUE(IONAME(SetRec)(io, j)) << "SetRec(" << j << ')';
    ASSERT_TRUE(IONAME(OutputInteger64)(io, j)) << "OutputInteger64()";
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for OutputInteger64";
  }

  for (int j{records}; j >= 1; --j) {
    // READ(UNIT=unit,FMT=fmt,REC=j) n
    io = IONAME(BeginExternalFormattedInput)(
        fmt, sizeof fmt - 1, nullptr, unit, __FILE__, __LINE__);
    ASSERT_TRUE(IONAME(SetRec)(io, j)) << "SetRec(" << j << ')';
    std::int64_t buffer;
    ASSERT_TRUE(IONAME(InputInteger)(io, buffer)) << "InputInteger()";
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for InputInteger";

    ASSERT_EQ(buffer, j) << "Read back " << buffer
                         << " from direct formatted record " << j
                         << ", expected " << j << '\n';
  }

  // CLOSE(UNIT=unit,STATUS='DELETE')
  io = IONAME(BeginClose)(unit, __FILE__, __LINE__);
  ASSERT_TRUE(IONAME(SetStatus)(io, "DELETE", 6)) << "SetStatus(DELETE)";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for Close";
}

TEST(ExternalIOTests, TestSequentialVariableFormatted) {
  // OPEN(NEWUNIT=unit,ACCESS='SEQUENTIAL',ACTION='READWRITE',&
  //   FORM='FORMATTED',STATUS='SCRATCH')
  auto *io{IONAME(BeginOpenNewUnit)(__FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(SetAccess)(io, "SEQUENTIAL", 10))
      << "SetAccess(SEQUENTIAL)";
  ASSERT_TRUE(IONAME(SetAction)(io, "READWRITE", 9)) << "SetAction(READWRITE)";
  ASSERT_TRUE(IONAME(SetForm)(io, "FORMATTED", 9)) << "SetForm(FORMATTED)";
  ASSERT_TRUE(IONAME(SetStatus)(io, "SCRATCH", 7)) << "SetStatus(SCRATCH)";

  int unit{-1};
  ASSERT_TRUE(IONAME(GetNewUnit)(io, unit)) << "GetNewUnit()";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for OpenNewUnit";

  static const int records{10};
  std::int64_t buffer[records]; // INTEGER*8 :: BUFFER(0:9) = [(j,j=0,9)]
  for (int j{0}; j < records; ++j) {
    buffer[j] = j;
  }

  char fmt[32];
  for (int j{1}; j <= records; ++j) {
    std::snprintf(fmt, sizeof fmt, "(%dI4)", j);
    // DO J=1,RECORDS; WRITE(UNIT=unit,FMT=fmt) BUFFER(0:j); END DO
    io = IONAME(BeginExternalFormattedOutput)(
        fmt, std::strlen(fmt), nullptr, unit, __FILE__, __LINE__);
    for (int k{0}; k < j; ++k) {
      ASSERT_TRUE(IONAME(OutputInteger64)(io, buffer[k]))
          << "OutputInteger64()";
    }
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for OutputInteger64";
  }

  // REWIND(UNIT=unit)
  io = IONAME(BeginRewind)(unit, __FILE__, __LINE__);
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for Rewind";

  for (int j{1}; j <= records; ++j) {
    std::snprintf(fmt, sizeof fmt, "(%dI4)", j);
    // DO J=1,RECORDS; READ(UNIT=unit,FMT=fmt) n; check n; END DO
    io = IONAME(BeginExternalFormattedInput)(
        fmt, std::strlen(fmt), nullptr, unit, __FILE__, __LINE__);

    std::int64_t check[records];
    for (int k{0}; k < j; ++k) {
      ASSERT_TRUE(IONAME(InputInteger)(io, check[k])) << "InputInteger()";
    }
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for InputInteger";

    for (int k{0}; k < j; ++k) {
      ASSERT_EQ(buffer[k], check[k])
          << "Read back [" << k << "]=" << check[k]
          << " from sequential variable formatted record " << j << ", expected "
          << buffer[k] << '\n';
    }
  }

  for (int j{records}; j >= 1; --j) {
    // BACKSPACE(unit)
    io = IONAME(BeginBackspace)(unit, __FILE__, __LINE__);
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for Backspace (before read)";

    std::snprintf(fmt, sizeof fmt, "(%dI4)", j);
    // READ(UNIT=unit,FMT=fmt,SIZE=chars) n; check
    io = IONAME(BeginExternalFormattedInput)(
        fmt, std::strlen(fmt), nullptr, unit, __FILE__, __LINE__);

    std::int64_t check[records];
    for (int k{0}; k < j; ++k) {
      ASSERT_TRUE(IONAME(InputInteger)(io, check[k])) << "InputInteger()";
    }

    std::size_t chars{IONAME(GetSize)(io)};
    ASSERT_EQ(chars, j * 4u)
        << "GetSize()=" << chars << ", expected " << (j * 4u) << '\n';
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for InputInteger";
    for (int k{0}; k < j; ++k) {
      ASSERT_EQ(buffer[k], check[k])
          << "Read back [" << k << "]=" << buffer[k]
          << " from sequential variable formatted record " << j << ", expected "
          << buffer[k] << '\n';
    }

    // BACKSPACE(unit)
    io = IONAME(BeginBackspace)(unit, __FILE__, __LINE__);
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for Backspace (after read)";
  }

  // CLOSE(UNIT=unit,STATUS='DELETE')
  io = IONAME(BeginClose)(unit, __FILE__, __LINE__);
  ASSERT_TRUE(IONAME(SetStatus)(io, "DELETE", 6)) << "SetStatus(DELETE)";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for Close";
}

TEST(ExternalIOTests, TestNonAvancingInput) {
  // OPEN(NEWUNIT=unit,ACCESS='SEQUENTIAL',ACTION='READWRITE',&
  //   FORM='FORMATTED',STATUS='SCRATCH')
  auto *io{IONAME(BeginOpenNewUnit)(__FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(SetAccess)(io, "SEQUENTIAL", 10))
      << "SetAccess(SEQUENTIAL)";
  ASSERT_TRUE(IONAME(SetAction)(io, "READWRITE", 9)) << "SetAction(READWRITE)";
  ASSERT_TRUE(IONAME(SetForm)(io, "FORMATTED", 9)) << "SetForm(FORMATTED)";
  ASSERT_TRUE(IONAME(SetStatus)(io, "SCRATCH", 7)) << "SetStatus(SCRATCH)";

  int unit{-1};
  ASSERT_TRUE(IONAME(GetNewUnit)(io, unit)) << "GetNewUnit()";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for OpenNewUnit";

  // Write the file to be used for the input test.
  static constexpr std::string_view records[] = {
      "ABCDEFGH", "IJKLMNOP", "QRSTUVWX"};
  static constexpr std::string_view fmt{"(A)"};
  for (const auto &record : records) {
    // WRITE(UNIT=unit,FMT=fmt) record
    io = IONAME(BeginExternalFormattedOutput)(
        fmt.data(), fmt.length(), nullptr, unit, __FILE__, __LINE__);
    ASSERT_TRUE(IONAME(OutputAscii)(io, record.data(), record.length()))
        << "OutputAscii()";
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for OutputAscii";
  }

  // REWIND(UNIT=unit)
  io = IONAME(BeginRewind)(unit, __FILE__, __LINE__);
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for Rewind";

  struct TestItems {
    std::string item;
    int expectedIoStat;
    std::string expectedItemValue[2];
  };
  // Actual non advancing input IO test
  TestItems inputItems[]{
      {std::string(4, '+'), IostatOk, {"ABCD", "ABCD"}},
      {std::string(4, '+'), IostatOk, {"EFGH", "EFGH"}},
      {std::string(4, '+'), IostatEor, {"++++", "    "}},
      {std::string(2, '+'), IostatOk, {"IJ", "IJ"}},
      {std::string(8, '+'), IostatEor, {"++++++++", "KLMNOP  "}},
      {std::string(10, '+'), IostatEor, {"++++++++++", "QRSTUVWX  "}},
  };

  // Test with PAD='NO'
  int j{0};
  for (auto &inputItem : inputItems) {
    // READ(UNIT=unit, FMT=fmt, ADVANCE='NO', PAD='NO', IOSTAT=iostat) inputItem
    io = IONAME(BeginExternalFormattedInput)(
        fmt.data(), fmt.length(), nullptr, unit, __FILE__, __LINE__);
    IONAME(EnableHandlers)(io, true, false, false, false, false);
    ASSERT_TRUE(IONAME(SetAdvance)(io, "NO", 2)) << "SetAdvance(NO)" << j;
    ASSERT_TRUE(IONAME(SetPad)(io, "NO", 2)) << "SetPad(NO)" << j;
    bool result{
        IONAME(InputAscii)(io, inputItem.item.data(), inputItem.item.length())};
    ASSERT_EQ(result, inputItem.expectedIoStat == IostatOk)
        << "InputAscii() " << j;
    ASSERT_EQ(IONAME(EndIoStatement)(io), inputItem.expectedIoStat)
        << "EndIoStatement() for Read " << j;
    ASSERT_EQ(inputItem.item, inputItem.expectedItemValue[0])
        << "Input-item value after non advancing read " << j;
    j++;
  }

  // REWIND(UNIT=unit)
  io = IONAME(BeginRewind)(unit, __FILE__, __LINE__);
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for Rewind";

  // Test again with PAD='YES'
  j = 0;
  for (auto &inputItem : inputItems) {
    // READ(UNIT=unit, FMT=fmt, ADVANCE='NO', PAD='YES', IOSTAT=iostat)
    // inputItem
    io = IONAME(BeginExternalFormattedInput)(
        fmt.data(), fmt.length(), nullptr, unit, __FILE__, __LINE__);
    IONAME(EnableHandlers)(io, true, false, false, false, false);
    ASSERT_TRUE(IONAME(SetAdvance)(io, "NO", 2)) << "SetAdvance(NO)" << j;
    ASSERT_TRUE(IONAME(SetPad)(io, "YES", 3)) << "SetPad(YES)" << j;
    bool result{
        IONAME(InputAscii)(io, inputItem.item.data(), inputItem.item.length())};
    ASSERT_EQ(result, inputItem.expectedIoStat == IostatOk)
        << "InputAscii() " << j;
    ASSERT_EQ(IONAME(EndIoStatement)(io), inputItem.expectedIoStat)
        << "EndIoStatement() for Read " << j;
    ASSERT_EQ(inputItem.item, inputItem.expectedItemValue[1])
        << "Input-item value after non advancing read " << j;
    j++;
  }

  // CLOSE(UNIT=unit)
  io = IONAME(BeginClose)(unit, __FILE__, __LINE__);
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for Close";
}

TEST(ExternalIOTests, TestWriteAfterNonAvancingInput) {
  // OPEN(NEWUNIT=unit,ACCESS='SEQUENTIAL',ACTION='READWRITE',&
  //   FORM='FORMATTED',STATUS='SCRATCH')
  auto *io{IONAME(BeginOpenNewUnit)(__FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(SetAccess)(io, "SEQUENTIAL", 10))
      << "SetAccess(SEQUENTIAL)";
  ASSERT_TRUE(IONAME(SetAction)(io, "READWRITE", 9)) << "SetAction(READWRITE)";
  ASSERT_TRUE(IONAME(SetForm)(io, "FORMATTED", 9)) << "SetForm(FORMATTED)";
  ASSERT_TRUE(IONAME(SetStatus)(io, "SCRATCH", 7)) << "SetStatus(SCRATCH)";

  int unit{-1};
  ASSERT_TRUE(IONAME(GetNewUnit)(io, unit)) << "GetNewUnit()";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for OpenNewUnit";

  // Write the file to be used for the input test.
  static constexpr std::string_view records[] = {"ABCDEFGHIJKLMNOPQRST"};
  static constexpr std::string_view fmt{"(A)"};
  for (const auto &record : records) {
    // WRITE(UNIT=unit,FMT=fmt) record
    io = IONAME(BeginExternalFormattedOutput)(
        fmt.data(), fmt.length(), nullptr, unit, __FILE__, __LINE__);
    ASSERT_TRUE(IONAME(OutputAscii)(io, record.data(), record.length()))
        << "OutputAscii()";
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for OutputAscii";
  }

  // REWIND(UNIT=unit)
  io = IONAME(BeginRewind)(unit, __FILE__, __LINE__);
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for Rewind";

  struct TestItems {
    std::string item;
    int expectedIoStat;
    std::string expectedItemValue;
  };
  // Actual non advancing input IO test
  TestItems inputItems[]{
      {std::string(4, '+'), IostatOk, "ABCD"},
      {std::string(4, '+'), IostatOk, "EFGH"},
  };

  int j{0};
  for (auto &inputItem : inputItems) {
    // READ(UNIT=unit, FMT=fmt, ADVANCE='NO', IOSTAT=iostat) inputItem
    io = IONAME(BeginExternalFormattedInput)(
        fmt.data(), fmt.length(), nullptr, unit, __FILE__, __LINE__);
    IONAME(EnableHandlers)(io, true, false, false, false, false);
    ASSERT_TRUE(IONAME(SetAdvance)(io, "NO", 2)) << "SetAdvance(NO)" << j;
    ASSERT_TRUE(
        IONAME(InputAscii)(io, inputItem.item.data(), inputItem.item.length()))
        << "InputAscii() " << j;
    ASSERT_EQ(IONAME(EndIoStatement)(io), inputItem.expectedIoStat)
        << "EndIoStatement() for Read " << j;
    ASSERT_EQ(inputItem.item, inputItem.expectedItemValue)
        << "Input-item value after non advancing read " << j;
    j++;
  }

  // WRITE(UNIT=unit, FMT=fmt, IOSTAT=iostat) outputItem.
  static constexpr std::string_view outputItem{"XYZ"};
  // WRITE(UNIT=unit,FMT=fmt) record
  io = IONAME(BeginExternalFormattedOutput)(
      fmt.data(), fmt.length(), nullptr, unit, __FILE__, __LINE__);
  ASSERT_TRUE(IONAME(OutputAscii)(io, outputItem.data(), outputItem.length()))
      << "OutputAscii()";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for OutputAscii";

  // Verify that the output was written in the record read in non advancing
  // mode, after the read part, and that the end was truncated.

  // REWIND(UNIT=unit)
  io = IONAME(BeginRewind)(unit, __FILE__, __LINE__);
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for Rewind";

  std::string resultRecord(20, '+');
  std::string expectedRecord{"ABCDEFGHXYZ         "};
  // READ(UNIT=unit, FMT=fmt, IOSTAT=iostat) result
  io = IONAME(BeginExternalFormattedInput)(
      fmt.data(), fmt.length(), nullptr, unit, __FILE__, __LINE__);
  IONAME(EnableHandlers)(io, true, false, false, false, false);
  ASSERT_TRUE(
      IONAME(InputAscii)(io, resultRecord.data(), resultRecord.length()))
      << "InputAscii() ";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for Read ";
  ASSERT_EQ(resultRecord, expectedRecord)
      << "Record after non advancing read followed by write";
  // CLOSE(UNIT=unit)
  io = IONAME(BeginClose)(unit, __FILE__, __LINE__);
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for Close";
}

TEST(ExternalIOTests, TestWriteAfterEndfile) {
  // OPEN(NEWUNIT=unit,ACCESS='SEQUENTIAL',ACTION='READWRITE',&
  //   FORM='FORMATTED',STATUS='SCRATCH')
  auto *io{IONAME(BeginOpenNewUnit)(__FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(SetAccess)(io, "SEQUENTIAL", 10))
      << "SetAccess(SEQUENTIAL)";
  ASSERT_TRUE(IONAME(SetAction)(io, "READWRITE", 9)) << "SetAction(READWRITE)";
  ASSERT_TRUE(IONAME(SetForm)(io, "FORMATTED", 9)) << "SetForm(FORMATTED)";
  ASSERT_TRUE(IONAME(SetStatus)(io, "SCRATCH", 7)) << "SetStatus(SCRATCH)";
  int unit{-1};
  ASSERT_TRUE(IONAME(GetNewUnit)(io, unit)) << "GetNewUnit()";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for OpenNewUnit";
  // WRITE(unit,"(I8)") 1234
  static constexpr std::string_view format{"(I8)"};
  io = IONAME(BeginExternalFormattedOutput)(
      format.data(), format.length(), nullptr, unit, __FILE__, __LINE__);
  ASSERT_TRUE(IONAME(OutputInteger64)(io, 1234)) << "OutputInteger64()";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement for WRITE before ENDFILE";
  // ENDFILE(unit)
  io = IONAME(BeginEndfile)(unit, __FILE__, __LINE__);
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement for ENDFILE";
  // WRITE(unit,"(I8)",iostat=iostat) 5678
  io = IONAME(BeginExternalFormattedOutput)(
      format.data(), format.length(), nullptr, unit, __FILE__, __LINE__);
  IONAME(EnableHandlers)(io, true /*IOSTAT=*/);
  ASSERT_FALSE(IONAME(OutputInteger64)(io, 5678)) << "OutputInteger64()";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatWriteAfterEndfile)
      << "EndIoStatement for WRITE after ENDFILE";
  // BACKSPACE(unit)
  io = IONAME(BeginBackspace)(unit, __FILE__, __LINE__);
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement for BACKSPACE";
  // WRITE(unit,"(I8)") 3456
  io = IONAME(BeginExternalFormattedOutput)(
      format.data(), format.length(), nullptr, unit, __FILE__, __LINE__);
  ASSERT_TRUE(IONAME(OutputInteger64)(io, 3456)) << "OutputInteger64()";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement for WRITE after BACKSPACE";
  // REWIND(unit)
  io = IONAME(BeginRewind)(unit, __FILE__, __LINE__);
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement for REWIND";
  // READ(unit,"(I8)",END=) j, k
  std::int64_t j{-1}, k{-1}, eof{-1};
  io = IONAME(BeginExternalFormattedInput)(
      format.data(), format.length(), nullptr, unit, __FILE__, __LINE__);
  IONAME(EnableHandlers)(io, false, false, true /*END=*/);
  ASSERT_TRUE(IONAME(InputInteger)(io, j)) << "InputInteger(j)";
  ASSERT_EQ(j, 1234) << "READ(j)";
  ASSERT_TRUE(IONAME(InputInteger)(io, k)) << "InputInteger(k)";
  ASSERT_EQ(k, 3456) << "READ(k)";
  ASSERT_FALSE(IONAME(InputInteger)(io, eof)) << "InputInteger(eof)";
  ASSERT_EQ(eof, -1) << "READ(eof)";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatEnd) << "EndIoStatement for READ";
  // CLOSE(UNIT=unit)
  io = IONAME(BeginClose)(unit, __FILE__, __LINE__);
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for Close";
}

TEST(ExternalIOTests, TestUTF8Encoding) {
  // OPEN(FILE="utf8test",NEWUNIT=unit,ACCESS='SEQUENTIAL',ACTION='READWRITE',&
  //   FORM='FORMATTED',STATUS='REPLACE',ENCODING='UTF-8')
  auto *io{IONAME(BeginOpenNewUnit)(__FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(SetAccess)(io, "SEQUENTIAL", 10))
      << "SetAccess(SEQUENTIAL)";
  ASSERT_TRUE(IONAME(SetAction)(io, "READWRITE", 9)) << "SetAction(READWRITE)";
  ASSERT_TRUE(IONAME(SetFile)(io, "utf8test", 8)) << "SetFile(utf8test)";
  ASSERT_TRUE(IONAME(SetForm)(io, "FORMATTED", 9)) << "SetForm(FORMATTED)";
  ASSERT_TRUE(IONAME(SetStatus)(io, "REPLACE", 7)) << "SetStatus(REPLACE)";
  ASSERT_TRUE(IONAME(SetEncoding)(io, "UTF-8", 5)) << "SetEncoding(UTF-8)";
  int unit{-1};
  ASSERT_TRUE(IONAME(GetNewUnit)(io, unit)) << "GetNewUnit()";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for first OPEN";
  char buffer[12];
  std::memcpy(buffer,
      "abc\x80\xff"
      "de\0\0\0\0\0",
      12);
  // WRITE(unit, *) buffer
  io = IONAME(BeginExternalListOutput)(unit, __FILE__, __LINE__);
  StaticDescriptor<0> staticDescriptor;
  Descriptor &desc{staticDescriptor.descriptor()};
  desc.Establish(TypeCode{CFI_type_char}, 7, buffer, 0);
  desc.Check();
  ASSERT_TRUE(IONAME(OutputDescriptor)(io, desc));
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for WRITE";
  // REWIND(unit)
  io = IONAME(BeginRewind)(unit, __FILE__, __LINE__);
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement for REWIND";
  // READ(unit, *) buffer
  desc.Establish(TypeCode(CFI_type_char), sizeof buffer, buffer, 0);
  desc.Check();
  io = IONAME(BeginExternalListInput)(unit, __FILE__, __LINE__);
  ASSERT_TRUE(IONAME(InputDescriptor)(io, desc));
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for first READ";
  ASSERT_EQ(std::memcmp(buffer,
                "abc\x80\xff"
                "de     ",
                12),
      0);
  // CLOSE(UNIT=unit,STATUS='KEEP')
  io = IONAME(BeginClose)(unit, __FILE__, __LINE__);
  ASSERT_TRUE(IONAME(SetStatus)(io, "KEEP", 4)) << "SetStatus(KEEP)";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for first CLOSE";
  // OPEN(FILE="utf8test",NEWUNIT=unit,ACCESS='SEQUENTIAL',ACTION='READWRITE',&
  //   FORM='FORMATTED',STATUS='OLD')
  io = IONAME(BeginOpenNewUnit)(__FILE__, __LINE__);
  ASSERT_TRUE(IONAME(SetAccess)(io, "SEQUENTIAL", 10))
      << "SetAccess(SEQUENTIAL)";
  ASSERT_TRUE(IONAME(SetAction)(io, "READWRITE", 9)) << "SetAction(READWRITE)";
  ASSERT_TRUE(IONAME(SetFile)(io, "utf8test", 8)) << "SetFile(utf8test)";
  ASSERT_TRUE(IONAME(SetForm)(io, "FORMATTED", 9)) << "SetForm(FORMATTED)";
  ASSERT_TRUE(IONAME(SetStatus)(io, "OLD", 3)) << "SetStatus(OLD)";
  ASSERT_TRUE(IONAME(GetNewUnit)(io, unit)) << "GetNewUnit()";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for second OPEN";
  // READ(unit, *) buffer
  io = IONAME(BeginExternalListInput)(unit, __FILE__, __LINE__);
  ASSERT_TRUE(IONAME(InputDescriptor)(io, desc));
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for second READ";
  ASSERT_EQ(std::memcmp(buffer,
                "abc\xc2\x80\xc3\xbf"
                "de   ",
                12),
      0);
  // CLOSE(UNIT=unit,STATUS='DELETE')
  io = IONAME(BeginClose)(unit, __FILE__, __LINE__);
  ASSERT_TRUE(IONAME(SetStatus)(io, "DELETE", 6)) << "SetStatus(DELETE)";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for second CLOSE";
}

TEST(ExternalIOTests, TestUCS) {
  // OPEN(FILE="ucstest',NEWUNIT=unit,ACCESS='SEQUENTIAL',ACTION='READWRITE',&
  //   FORM='FORMATTED',STATUS='REPLACE',ENCODING='UTF-8')
  auto *io{IONAME(BeginOpenNewUnit)(__FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(SetAccess)(io, "SEQUENTIAL", 10))
      << "SetAccess(SEQUENTIAL)";
  ASSERT_TRUE(IONAME(SetAction)(io, "READWRITE", 9)) << "SetAction(READWRITE)";
  ASSERT_TRUE(IONAME(SetFile)(io, "ucstest", 7)) << "SetAction(ucstest)";
  ASSERT_TRUE(IONAME(SetForm)(io, "FORMATTED", 9)) << "SetForm(FORMATTED)";
  ASSERT_TRUE(IONAME(SetStatus)(io, "REPLACE", 7)) << "SetStatus(REPLACE)";
  ASSERT_TRUE(IONAME(SetEncoding)(io, "UTF-8", 5)) << "SetEncoding(UTF-8)";
  int unit{-1};
  ASSERT_TRUE(IONAME(GetNewUnit)(io, unit)) << "GetNewUnit()";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for first OPEN";
  char32_t wbuffer[8]{U"abc\u0080\uffff"
                      "de"};
  // WRITE(unit, *) wbuffec
  io = IONAME(BeginExternalListOutput)(unit, __FILE__, __LINE__);
  StaticDescriptor<0> staticDescriptor;
  Descriptor &desc{staticDescriptor.descriptor()};
  desc.Establish(TypeCode{CFI_type_char32_t}, sizeof wbuffer - sizeof(char32_t),
      wbuffer, 0);
  desc.Check();
  ASSERT_TRUE(IONAME(OutputDescriptor)(io, desc));
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for WRITE";
  // REWIND(unit)
  io = IONAME(BeginRewind)(unit, __FILE__, __LINE__);
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement for REWIND";
  // READ(unit, *) buffer
  io = IONAME(BeginExternalListInput)(unit, __FILE__, __LINE__);
  desc.Establish(TypeCode{CFI_type_char32_t}, sizeof wbuffer, wbuffer, 0);
  desc.Check();
  ASSERT_TRUE(IONAME(InputDescriptor)(io, desc));
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for first READ";
  char dump[80];
  dump[0] = '\0';
  for (int j{0}; j < 8; ++j) {
    std::size_t dumpLen{std::strlen(dump)};
    std::snprintf(
        dump + dumpLen, sizeof dump - dumpLen, " %x", (unsigned)wbuffer[j]);
  }
  EXPECT_EQ(wbuffer[0], U'a') << dump;
  EXPECT_EQ(wbuffer[1], U'b') << dump;
  EXPECT_EQ(wbuffer[2], U'c') << dump;
  EXPECT_EQ(wbuffer[3], U'\u0080') << dump;
  EXPECT_EQ(wbuffer[4], U'\uffff') << dump;
  EXPECT_EQ(wbuffer[5], U'd') << dump;
  EXPECT_EQ(wbuffer[6], U'e') << dump;
  EXPECT_EQ(wbuffer[7], U' ') << dump;
  // CLOSE(UNIT=unit,STATUS='KEEP')
  io = IONAME(BeginClose)(unit, __FILE__, __LINE__);
  ASSERT_TRUE(IONAME(SetStatus)(io, "KEEP", 4)) << "SetStatus(KEEP)";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for first CLOSE";
  // OPEN(FILE="ucstest",NEWUNIT=unit,ACCESS='SEQUENTIAL',ACTION='READWRITE',&
  //   FORM='FORMATTED',STATUS='OLD')
  io = IONAME(BeginOpenNewUnit)(__FILE__, __LINE__);
  ASSERT_TRUE(IONAME(SetAccess)(io, "SEQUENTIAL", 10))
      << "SetAccess(SEQUENTIAL)";
  ASSERT_TRUE(IONAME(SetAction)(io, "READWRITE", 9)) << "SetAction(READWRITE)";
  ASSERT_TRUE(IONAME(SetFile)(io, "ucstest", 7)) << "SetFile(ucstest)";
  ASSERT_TRUE(IONAME(SetForm)(io, "FORMATTED", 9)) << "SetForm(FORMATTED)";
  ASSERT_TRUE(IONAME(SetStatus)(io, "OLD", 3)) << "SetStatus(OLD)";
  ASSERT_TRUE(IONAME(GetNewUnit)(io, unit)) << "GetNewUnit()";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for second OPEN";
  char buffer[12];
  // READ(unit, *) buffer
  io = IONAME(BeginExternalListInput)(unit, __FILE__, __LINE__);
  desc.Establish(TypeCode{CFI_type_char}, sizeof buffer, buffer, 0);
  desc.Check();
  ASSERT_TRUE(IONAME(InputDescriptor)(io, desc));
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for second READ";
  dump[0] = '\0';
  for (int j{0}; j < 12; ++j) {
    std::size_t dumpLen{std::strlen(dump)};
    std::snprintf(dump + dumpLen, sizeof dump - dumpLen, " %x",
        (unsigned)(unsigned char)buffer[j]);
  }
  EXPECT_EQ(std::memcmp(buffer,
                "abc\xc2\x80\xef\xbf\xbf"
                "de  ",
                12),
      0)
      << dump;
  // CLOSE(UNIT=unit,STATUS='DELETE')
  io = IONAME(BeginClose)(unit, __FILE__, __LINE__);
  ASSERT_TRUE(IONAME(SetStatus)(io, "DELETE", 6)) << "SetStatus(DELETE)";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for second CLOSE";
}

TEST(ExternalIOTests, BigUnitNumbers) {
  if (std::numeric_limits<ExternalUnit>::max() <
      std::numeric_limits<std::int64_t>::max()) {
    std::int64_t unit64Ok = std::numeric_limits<ExternalUnit>::max();
    std::int64_t unit64Bad = unit64Ok + 1;
    std::int64_t unit64Bad2 =
        static_cast<std::int64_t>(std::numeric_limits<ExternalUnit>::min()) - 1;
    EXPECT_EQ(IONAME(CheckUnitNumberInRange64)(unit64Ok, true), IostatOk);
    EXPECT_EQ(IONAME(CheckUnitNumberInRange64)(unit64Ok, false), IostatOk);
    EXPECT_EQ(
        IONAME(CheckUnitNumberInRange64)(unit64Bad, true), IostatUnitOverflow);
    EXPECT_EQ(
        IONAME(CheckUnitNumberInRange64)(unit64Bad2, true), IostatUnitOverflow);
    constexpr std::size_t n{80};
    char expectedMsg[n + 1];
    expectedMsg[n] = '\0';
    std::snprintf(expectedMsg, n, "UNIT number %jd is out of range",
        static_cast<std::intmax_t>(unit64Bad));
    EXPECT_DEATH(
        IONAME(CheckUnitNumberInRange64)(unit64Bad, false), expectedMsg);
    for (auto i{std::strlen(expectedMsg)}; i < n; ++i) {
      expectedMsg[i] = ' ';
    }
    char msg[n + 1];
    msg[n] = '\0';
    EXPECT_EQ(IONAME(CheckUnitNumberInRange64)(unit64Bad, true, msg, n),
        IostatUnitOverflow);
    EXPECT_EQ(std::strncmp(msg, expectedMsg, n), 0);
  }
}
