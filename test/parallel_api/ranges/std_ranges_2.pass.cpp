// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#include "std_ranges_test.h"

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING

    using namespace test_std_ranges;

    test_range_algo{}(oneapi::dpl::ranges::count_if, std::ranges::count_if, pred, proj);
    test_range_algo{}(oneapi::dpl::ranges::count, std::ranges::count, 4, proj);

    test_range_algo{}(oneapi::dpl::ranges::is_sorted, std::ranges::is_sorted, comp, proj);
    test_range_algo<data_in_in>{}(oneapi::dpl::ranges::equal,  std::ranges::equal, pred_2, proj);

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
