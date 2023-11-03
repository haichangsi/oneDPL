// -*- C++ -*-
//===-- parallel_backend_sycl_reduce.h --------------------------------===//
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

#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_HISTOGRAM_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_HISTOGRAM_H

#include <algorithm>
#include <cstdint>
#include <type_traits>


#include "sycl_defs.h"
#include "parallel_backend_sycl_utils.h"
#include "execution_sycl_defs.h"
#include "unseq_backend_sycl.h"
#include "utils_ranges_sycl.h"
#include "oneapi/dpl/internal/async_impl/async_impl_hetero.h"

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

template <typename _HistAccessor, typename _OffsetT, typename _Size>
inline void
__clear_wglocal_histograms(const _HistAccessor& local_histogram, const _OffsetT& offset, const _Size& __num_bins,
                           const sycl::nd_item<1>& self_item)
{
    ::std::uint32_t gSize = self_item.get_local_range()[0];
    ::std::uint32_t self_lidx = self_item.get_local_id(0);
    ::std::uint8_t factor = oneapi::dpl::__internal::__dpl_ceiling_div(__num_bins, gSize);
    ::std::uint8_t k;
    // no need for atomicity when we are explicitly assigning work-items to
    // locations

    _ONEDPL_PRAGMA_UNROLL
    for (k = 0; k < factor - 1; k++)
    {
        local_histogram[offset + gSize * k + self_lidx] = 0;
    }
    // residual
    if (gSize * k + self_lidx < __num_bins)
    {
        local_histogram[offset + gSize * k + self_lidx] = 0;
    }
    __dpl_sycl::__group_barrier(self_item);
}

template <typename _BinIdxType, typename _Iter1, typename _HistReg, typename _BinFunc>
inline void
__accum_local_register_iter(const _Iter1& in_acc, const ::std::size_t& __index, _HistReg* __histogram, _BinFunc __func,
                            void* __boost_mem)
{
    const auto& x = in_acc[__index];
    if (__func.is_valid(x))
    {
        _BinIdxType c = __func.get_bin(x, __boost_mem);
        __histogram[c]++;
    }
}

template <typename _BinIdxType, sycl::access::address_space _AddressSpace, typename _Iter1, typename _HistAccessor,
          typename _OffsetT, typename _BinFunc, typename... _VoidType>
inline void
__accum_local_atomics_iter(const _Iter1& in_acc, const ::std::size_t& __index, const _HistAccessor& wg_local_histogram,
                           const _OffsetT& offset, _BinFunc func, _VoidType... __boost_mem)
{
    using __histo_value_type = typename _HistAccessor::value_type;
    const auto& x = in_acc[__index];
    if (func.is_valid(x))
    {
        _BinIdxType c = func.get_bin(x, __boost_mem...);
        __dpl_sycl::__atomic_ref<__histo_value_type, _AddressSpace> local_bin(wg_local_histogram[offset + c]);
        local_bin++;
    }
}

template <typename _BinType, typename _HistAccessorIn, typename _OffsetT, typename _HistAccessorOut, typename _Size>
inline void
__reduce_out_histograms(const _HistAccessorIn& in_histogram, const _OffsetT& offset,
                        const _HistAccessorOut& out_histogram, const _Size& __num_bins, const sycl::nd_item<1>& self_item)
{
    ::std::uint32_t gSize = self_item.get_local_range()[0];
    ::std::uint32_t self_lidx = self_item.get_local_id(0);
    ::std::uint8_t factor = oneapi::dpl::__internal::__dpl_ceiling_div(__num_bins, gSize);
    ::std::uint8_t k;

    _ONEDPL_PRAGMA_UNROLL
    for (k = 0; k < factor - 1; k++)
    {
        __dpl_sycl::__atomic_ref<_BinType, sycl::access::address_space::global_space> global_bin(
            out_histogram[gSize * k + self_lidx]);
        global_bin += in_histogram[offset + gSize * k + self_lidx];
    }
    // residual
    if (gSize * k + self_lidx < __num_bins)
    {
        __dpl_sycl::__atomic_ref<_BinType, sycl::access::address_space::global_space> global_bin(
            out_histogram[gSize * k + self_lidx]);
        global_bin += in_histogram[offset + gSize * k + self_lidx];
    }
}

template <::std::uint16_t __iters_per_work_item, ::std::uint8_t __bins_per_work_item, typename Policy, typename _Range1,
          typename _Range2, typename _Size, typename _IdxHashFunc, typename... _Range3>
inline void
__histogram_general_registers_local_reduction(Policy&& policy, const sycl::event& __init_e,
                                              ::std::uint16_t __work_group_size, _Range1&& __input, _Range2&& __bins,
                                              const _Size& __num_bins, _IdxHashFunc __func, _Range3&&... __opt_range)
{

    const ::std::size_t N = __input.size();
    using __local_histogram_type = ::std::uint32_t;
    using __private_histogram_type = ::std::uint16_t;
    using __histogram_index_type = ::std::uint8_t;
    using __bin_type = oneapi::dpl::__internal::__value_t<_Range2>;

    ::std::size_t extra =
        oneapi::dpl::__internal::__dpl_ceiling_div(__func.get_required_SLM_memory(), sizeof(__local_histogram_type));

    ::std::size_t segments = oneapi::dpl::__internal::__dpl_ceiling_div(N, __work_group_size * __iters_per_work_item);
    auto e = policy.queue().submit([&](auto& h) {
        h.depends_on(__init_e);
        oneapi::dpl::__ranges::__require_access(h, __input, __bins, __opt_range...);
        __dpl_sycl::__local_accessor<__local_histogram_type> local_histogram(sycl::range(__num_bins + extra), h);
        h.parallel_for(
            sycl::nd_range<1>(segments * __work_group_size, __work_group_size), [=](sycl::nd_item<1> __self_item) {
                const ::std::size_t __self_lidx = __self_item.get_local_id(0);
                const ::std::size_t __wgroup_idx = __self_item.get_group(0);
                const ::std::size_t __seg_start = __work_group_size * __iters_per_work_item * __wgroup_idx;
                void* boost_mem = (void*)(&(local_histogram[0]) + __num_bins);
                __func.init_SLM_memory(boost_mem, __self_item);
                __clear_wglocal_histograms(local_histogram, 0, __num_bins, __self_item);
                __private_histogram_type histogram[__bins_per_work_item];
                _ONEDPL_PRAGMA_UNROLL
                for (__histogram_index_type k = 0; k < __bins_per_work_item; k++)
                {
                    histogram[k] = 0;
                }

                if (__seg_start + __work_group_size * __iters_per_work_item < N)
                {
                    _ONEDPL_PRAGMA_UNROLL
                    for (__histogram_index_type idx = 0; idx < __iters_per_work_item; idx++)
                    {
                        __accum_local_register_iter<__histogram_index_type>(
                            __input, __seg_start + idx * __work_group_size + __self_lidx, histogram, __func, boost_mem);
                    }
                }
                else
                {
                    _ONEDPL_PRAGMA_UNROLL
                    for (__histogram_index_type idx = 0; idx < __iters_per_work_item; idx++)
                    {
                        ::std::size_t __val_idx = __seg_start + idx * __work_group_size + __self_lidx;
                        if (__val_idx < N)
                        {
                            __accum_local_register_iter<__histogram_index_type>(__input, __val_idx, histogram, __func,
                                                                                boost_mem);
                        }
                    }
                }

                _ONEDPL_PRAGMA_UNROLL
                for (__histogram_index_type k = 0; k < __num_bins; k++)
                {
                    __dpl_sycl::__atomic_ref<__local_histogram_type, sycl::access::address_space::local_space>
                        local_bin(local_histogram[k]);
                    local_bin += histogram[k];
                }

                __dpl_sycl::__group_barrier(__self_item);

                __reduce_out_histograms<__bin_type>(local_histogram, 0, __bins, __num_bins, __self_item);
            });
    });
    e.wait();
}

template <::std::uint16_t __iters_per_work_item, typename Policy, typename _Range1, typename _Range2, typename _Size,
          typename _IdxHashFunc, typename... _Range3>
inline void
__histogram_general_local_atomics(Policy&& policy, const sycl::event& __init_e, ::std::uint16_t __work_group_size,
                                  _Range1&& __input, _Range2&& __bins, const _Size& __num_bins, _IdxHashFunc __func,
                                  _Range3&&... __opt_range)
{

    using __local_histogram_type = ::std::uint32_t;
    using __bin_type = oneapi::dpl::__internal::__value_t<_Range2>;
    using __histogram_index_type = ::std::uint16_t;

    ::std::size_t extra =
        oneapi::dpl::__internal::__dpl_ceiling_div(__func.get_required_SLM_memory(), sizeof(__local_histogram_type));
    const ::std::size_t N = __input.size();
    std::size_t segments = oneapi::dpl::__internal::__dpl_ceiling_div(N, __work_group_size * __iters_per_work_item);
    auto e = policy.queue().submit([&](auto& h) {
        h.depends_on(__init_e);
        oneapi::dpl::__ranges::__require_access(h, __input, __bins, __opt_range...);
        // minimum type size for atomics
        __dpl_sycl::__local_accessor<__local_histogram_type> local_histogram(sycl::range(__num_bins + extra), h);
        h.parallel_for(sycl::nd_range<1>(segments * __work_group_size, __work_group_size),
                       [=](sycl::nd_item<1> __self_item) {
                           constexpr auto _atomic_address_space = sycl::access::address_space::local_space;
                           const ::std::size_t __self_lidx = __self_item.get_local_id(0);
                           const ::std::uint32_t __wgroup_idx = __self_item.get_group(0);
                           const ::std::size_t __seg_start = __work_group_size * __wgroup_idx * __iters_per_work_item;
                           void* boost_mem = (void*)(&(local_histogram[0]) + __num_bins);
                           __func.init_SLM_memory(boost_mem, __self_item);

                           __clear_wglocal_histograms(local_histogram, 0, __num_bins, __self_item);

                           if (__seg_start + __work_group_size * __iters_per_work_item < N)
                           {
                               _ONEDPL_PRAGMA_UNROLL
                               for (::std::uint8_t idx = 0; idx < __iters_per_work_item; idx++)
                               {
                                   ::std::size_t __val_idx = __seg_start + idx * __work_group_size + __self_lidx;
                                   __accum_local_atomics_iter<__histogram_index_type, _atomic_address_space>(
                                       __input, __val_idx, local_histogram, 0, __func, boost_mem);
                               }
                           }
                           else
                           {
                               _ONEDPL_PRAGMA_UNROLL
                               for (::std::uint8_t idx = 0; idx < __iters_per_work_item; idx++)
                               {
                                   ::std::size_t __val_idx = __seg_start + idx * __work_group_size + __self_lidx;
                                   if (__val_idx < N)
                                   {
                                       __accum_local_atomics_iter<__histogram_index_type, _atomic_address_space>(
                                           __input, __val_idx, local_histogram, 0, __func, boost_mem);
                                   }
                               }
                           }
                           __dpl_sycl::__group_barrier(__self_item);

                           __reduce_out_histograms<__bin_type>(local_histogram, 0, __bins, __num_bins, __self_item);
                       });
    });

    e.wait();
}

template <::std::uint16_t __min_iters_per_work_item, typename Policy, typename _Range1, typename _Range2,
          typename _Size, typename _IdxHashFunc, typename... _Range3>
inline void
__histogram_general_private_global_atomics(Policy&& policy, const sycl::event& __init_e,
                                           ::std::uint16_t __work_group_size, _Range1&& __input, _Range2&& __bins,
                                           const _Size& __num_bins, _IdxHashFunc __func, _Range3&&... __opt_range)
{
    const ::std::size_t N = __input.size();
    using __bin_type = oneapi::dpl::__internal::__value_t<_Range2>;
    using __histogram_index_type = ::std::uint32_t;

    auto __global_mem_size = policy.queue().get_device().template get_info<sycl::info::device::global_mem_size>();
    const ::std::size_t max_segments =
        ::std::min(__global_mem_size / (__num_bins * sizeof(__bin_type)),
                   oneapi::dpl::__internal::__dpl_ceiling_div(N, __work_group_size * __min_iters_per_work_item));
    const ::std::size_t iters_per_work_item =
        oneapi::dpl::__internal::__dpl_ceiling_div(N, max_segments * __work_group_size);
    ::std::size_t segments = oneapi::dpl::__internal::__dpl_ceiling_div(N, __work_group_size * iters_per_work_item);

    auto private_histograms =
        oneapi::dpl::__par_backend_hetero::__internal::__buffer<Policy, __bin_type>(policy, segments * __num_bins)
            .get_buffer();

    auto e = policy.queue().submit([&](auto& h) {
        h.depends_on(__init_e);
        oneapi::dpl::__ranges::__require_access(h, __input, __bins, __opt_range...);
        sycl::accessor hacc_private{private_histograms, h, sycl::read_write, sycl::no_init};
        h.parallel_for(sycl::nd_range<1>(segments * __work_group_size, __work_group_size),
                       [=](sycl::nd_item<1> __self_item) {
                           constexpr auto _atomic_address_space = sycl::access::address_space::global_space;
                           const ::std::size_t __self_lidx = __self_item.get_local_id(0);
                           const ::std::size_t __wgroup_idx = __self_item.get_group(0);
                           const ::std::size_t __seg_start = __work_group_size * iters_per_work_item * __wgroup_idx;

                           __clear_wglocal_histograms(hacc_private, __wgroup_idx * __num_bins, __num_bins, __self_item);

                           if (__seg_start + __work_group_size * iters_per_work_item < N)
                           {
                               for (::std::size_t idx = 0; idx < iters_per_work_item; idx++)
                               {
                                   ::std::size_t __val_idx = __seg_start + idx * __work_group_size + __self_lidx;
                                   __accum_local_atomics_iter<__histogram_index_type, _atomic_address_space>(
                                       __input, __val_idx, hacc_private, __wgroup_idx * __num_bins, __func);
                               }
                           }
                           else
                           {
                               for (::std::size_t idx = 0; idx < iters_per_work_item; idx++)
                               {
                                   ::std::size_t __val_idx = __seg_start + idx * __work_group_size + __self_lidx;
                                   if (__val_idx < N)
                                   {
                                       __accum_local_atomics_iter<__histogram_index_type, _atomic_address_space>(
                                           __input, __val_idx, hacc_private, __wgroup_idx * __num_bins, __func);
                                   }
                               }
                           }

                           __dpl_sycl::__group_barrier(__self_item);

                           __reduce_out_histograms<__bin_type>(hacc_private, __wgroup_idx * __num_bins, __bins, __num_bins,
                                                               __self_item);
                       });
    });
    e.wait();
}

template <::std::uint16_t __iters_per_work_item, typename Policy, typename _Iter1, typename _Iter2, typename _Size,
          typename _IdxHashFunc, typename... _Range>
inline void
__parallel_histogram_impl(Policy&& policy, _Iter1 __first, _Iter1 __last, _Iter2 __histogram_first,
                          const _Size& __num_bins, _IdxHashFunc __func, _Range&&... __opt_range)
{
    using __local_histogram_type = ::std::uint32_t;
    using __global_histogram_type = typename ::std::iterator_traits<_Iter2>::value_type;

    ::std::size_t __max_wgroup_size = oneapi::dpl::__internal::__max_work_group_size(policy);

    ::std::uint16_t __work_group_size = ::std::min(::std::size_t(1024), __max_wgroup_size);

    auto __local_mem_size = policy.queue().get_device().template get_info<sycl::info::device::local_mem_size>();
    constexpr ::std::uint8_t __max_work_item_private_bins = 16;


    auto keep_bins =
        oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::write, _Iter2>();
    auto bins_buf = keep_bins(__histogram_first, __histogram_first + __num_bins);

    auto __f = oneapi::dpl::__internal::fill_functor<__global_histogram_type>{__global_histogram_type{0}};
    //fill histogram bins with zeros
    auto init_e = oneapi::dpl::__par_backend_hetero::__parallel_for(
        ::std::forward<Policy>(policy), unseq_backend::walk_n<Policy, decltype(__f)>{__f}, __num_bins,
        bins_buf.all_view());
    auto N = __last - __first;

    if (N > 0)
    {
        auto keep_input =
            oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::read, _Iter1>();
        auto input_buf = keep_input(__first, __last);

        ::std::size_t req_SLM_bytes = __func.get_required_SLM_memory();
        ::std::size_t extra_SLM_elements =
            oneapi::dpl::__internal::__dpl_ceiling_div(req_SLM_bytes, sizeof(__local_histogram_type));
        // if bins fit into registers, use register private accumulation
        if (__num_bins < __max_work_item_private_bins)
        {
            __histogram_general_registers_local_reduction<__iters_per_work_item, 16>(
                ::std::forward<Policy>(policy), init_e, __work_group_size, input_buf.all_view(), bins_buf.all_view(),
                __num_bins, __func, std::forward<_Range...>(__opt_range)...);
        }
        // if bins fit into SLM, use local atomics
        else if ((__num_bins + extra_SLM_elements) * sizeof(__local_histogram_type) < __local_mem_size)
        {
            __histogram_general_local_atomics<__iters_per_work_item>(
                ::std::forward<Policy>(policy), init_e, __work_group_size, input_buf.all_view(), bins_buf.all_view(),
                __num_bins, __func, std::forward<_Range...>(__opt_range)...);
        }
        else // otherwise, use global atomics (private copies per workgroup)
        {
            __histogram_general_private_global_atomics<__iters_per_work_item>(
                ::std::forward<Policy>(policy), init_e, __work_group_size, input_buf.all_view(), bins_buf.all_view(),
                __num_bins, __func, std::forward<_Range...>(__opt_range)...);
        }
    }
    else
    {
        init_e.wait();
    }
    return __histogram_first + num_bins;
}

template <typename Policy, typename _Iter1, typename _Iter2, typename _Size, typename _IdxHashFunc, typename... _Range>
inline void
__parallel_histogram(Policy&& policy, _Iter1 __first, _Iter1 __last, _Iter2 __histogram_first, const _Size& __num_bins,
                     _IdxHashFunc __func, _Range&&... __opt_range)
{
    auto N = __last - __first;

    if (N <= 524288)
    {
        __parallel_histogram_impl</*iters_per_workitem = */ 4>(::std::forward<Policy>(policy), __first, __last,
                                                                      __histogram_first, __num_bins, __func,
                                                                      std::forward<_Range...>(__opt_range)...);
    }
    else
    {
        __parallel_histogram_impl</*iters_per_workitem = */ 32>(::std::forward<Policy>(policy), __first, __last,
                                                                       __histogram_first, __num_bins, __func,
                                                                       std::forward<_Range...>(__opt_range)...);
    }
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_HISTOGRAM_H