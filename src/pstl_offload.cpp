// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <new>
#include <cassert>
#include <cstdint>
#include <sycl/sycl.hpp>

#include <pstl_offload/internal/usm_memory_replacement_common.h>

#define _PSTL_OFFLOAD_BINARY_VERSION_MAJOR 1
#define _PSTL_OFFLOAD_BINARY_VERSION_MINOR 0
#define _PSTL_OFFLOAD_BINARY_VERSION_PATCH 0

#if _WIN64

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#include <detours.h>
#pragma GCC diagnostic pop

#endif

namespace __pstl_offload
{

using __free_func_type = void (*)(void*);

#if __linux__

// list of objects for delayed releasing
struct __delayed_free_list {
    __delayed_free_list* _M_next;
    void*                _M_to_free;
};

// are we inside dlsym call?
static thread_local bool __dlsym_called = false;
// objects released inside of dlsym call
static thread_local __delayed_free_list* __delayed_free = nullptr;

static void
__free_delayed_list(void* __ptr_to_free, __free_func_type __orig_free)
{
    // It's enough to check __delayed_free only at this point,
    // as __delayed_free filled only inside dlsym(RTLD_NEXT, "free").
    while (__delayed_free)
    {
        __delayed_free_list* __next = __delayed_free->_M_next;
        // it's possible that an object to be released during 1st call of __internal_free
        // would be released 2nd time from inside nested dlsym call. To prevent "double free"
        // situation, check for it explicitly.
        if (__ptr_to_free != __delayed_free->_M_to_free)
        {
            __orig_free(__delayed_free->_M_to_free);
        }
        __orig_free(__delayed_free);
        __delayed_free = __next;
    }
}

static __free_func_type
__get_original_free_checked(void* __ptr_to_free)
{
    __dlsym_called = true;
    __free_func_type __orig_free = __free_func_type(dlsym(RTLD_NEXT, "free"));
    __dlsym_called = false;
    if (!__orig_free)
    {
        throw std::system_error(std::error_code(), dlerror());
    }

    // Releasing objects from delayed release list.
    __free_delayed_list(__ptr_to_free, __orig_free);

    return __orig_free;
}

static void
__original_free(void* __ptr_to_free)
{
    static __free_func_type __orig_free = __get_original_free_checked(__ptr_to_free);
    __orig_free(__ptr_to_free);
}

static std::size_t
__original_msize(void* __user_ptr)
{
    using __msize_func_type = std::size_t (*)(void*);

    static __msize_func_type __orig_msize =
        __msize_func_type(dlsym(RTLD_NEXT, "malloc_usable_size"));
    return __orig_msize(__user_ptr);
}

static void
__internal_free(void* __user_ptr)
{
    if (__user_ptr != nullptr)
    {
        __block_header* __header = static_cast<__block_header*>(__user_ptr) - 1;

        if (__same_memory_page(__user_ptr, __header) && __header->_M_uniq_const == __uniq_type_const)
        {
            __free_usm_pointer(__header);
        }
        else
        {
            if (__dlsym_called)
            {
                // Delay releasing till exit of dlsym. We do not overload malloc globally,
                // so can use it safely. Do not use new to able to use free() during
                // __delayed_free_list releasing.
                void* __buf = malloc(sizeof(__delayed_free_list));
                if (!__buf)
                {
                    throw std::bad_alloc();
                }
                __delayed_free_list* __h = new(__buf) __delayed_free_list{__delayed_free, __user_ptr};
                __delayed_free = __h;
            }
            else
            {
                __original_free(__user_ptr);
            }
        }
    }
}

#elif _WIN64

using __malloc_func_type = void* (*)(std::size_t);

static __free_func_type __original_free = free;
#if _DEBUG
static void (*__original_free_dbg)(void* userData, int blockType) = _free_dbg;
#endif
static __realloc_func_type __original_realloc = realloc;
static __free_func_type __original_aligned_free = _aligned_free;
static size_t (*__original_msize)(void *) = _msize;
static size_t (*__original_aligned_msize)(void *, std::size_t alignment, std::size_t offset) = _aligned_msize;
static void* (*__original_aligned_realloc)(void *, std::size_t size, std::size_t alignment) = _aligned_realloc;

static void
__internal_free_param(void* __user_ptr, __free_func_type __custom_free)
{
    if (__user_ptr != nullptr)
    {
        __block_header* __header = static_cast<__block_header*>(__user_ptr) - 1;

        if (__same_memory_page(__user_ptr, __header) && __header->_M_uniq_const == __uniq_type_const)
        {
            __free_usm_pointer(__header);
        }
        else
        {
            __custom_free(__user_ptr);
        }
    }
}

static void
__internal_free(void* __user_ptr)
{
    __internal_free_param(__user_ptr, __original_free);
}

static auto
__get_original_msize()
{
    return __original_msize;
}

#endif // _WIN64

static std::size_t
__internal_msize(void* __user_ptr)
{
    std::size_t __res = 0;
    if (__user_ptr != nullptr)
    {

        __block_header* __header = static_cast<__block_header*>(__user_ptr) - 1;

        if (__same_memory_page(__user_ptr, __header) && __header->_M_uniq_const == __uniq_type_const)
        {
            __res = __header->_M_requested_number_of_bytes;
        }
        else
        {
            __res = __original_msize(__user_ptr);
        }
    }
    return __res;
}

#if _WIN64

static std::size_t
__internal_aligned_msize(void* __user_ptr, std::size_t alignment, std::size_t offset)
{
    std::size_t __res = 0;
    if (__user_ptr != nullptr)
    {
        __block_header* __header = static_cast<__block_header*>(__user_ptr) - 1;

        if (__same_memory_page(__user_ptr, __header) && __header->_M_uniq_const == __uniq_type_const)
        {
            __res = __header->_M_requested_number_of_bytes;
        }
        else
        {
            __res = __original_aligned_msize(__user_ptr, alignment, offset);
        }
    }
    return __res;
}

static void
__internal_aligned_free(void* __user_ptr)
{
    __internal_free_param(__user_ptr, __original_aligned_free);
}

void* __aligned_realloc_real_pointer(void* __user_ptr, std::size_t __new_size, std::size_t __alignment)
{
    assert(__user_ptr != nullptr);

    if (!__new_size)
    {
        __internal_aligned_free(__user_ptr);
        return nullptr;
    }

    __block_header* __header = static_cast<__block_header*>(__user_ptr) - 1;

    void* __result = nullptr;

    if (__same_memory_page(__user_ptr, __header) && __header->_M_uniq_const == __uniq_type_const)
    {
        if (__header->_M_requested_number_of_bytes == __new_size && (uintptr_t)__user_ptr % __alignment == 0)
        {
            __result = __user_ptr;
        }
        else
        {
            assert(__header->_M_device != nullptr);
            void* __new_ptr = __allocate_shared_for_device(__header->_M_device, __new_size, __alignment);

            if (__new_ptr != nullptr)
            {
                std::memcpy(__new_ptr, __user_ptr, std::min(__header->_M_requested_number_of_bytes, __new_size));

                // Free previously allocated memory
                __free_usm_pointer(__header);
                __result = __new_ptr;
            }
            else
            {
                errno = ENOMEM;
            }
        }
    }
    else
    {
        // __user_ptr is not a USM pointer, use original realloc function
        __result = __original_aligned_realloc(__user_ptr, __new_size, __alignment);
    }
    return __result;
}

#if _DEBUG

static void
__internal_free_dbg(void* __user_ptr, int __type)
{
    if (__user_ptr != nullptr)
    {
        __block_header* __header = static_cast<__block_header*>(__user_ptr) - 1;

        if (__same_memory_page(__user_ptr, __header) && __header->_M_uniq_const == __uniq_type_const)
        {
            __free_usm_pointer(__header);
        }
        else
        {
            __original_free_dbg(__user_ptr, __type);
        }
    }
}

#endif // _DEBUG

__malloc_func_type
__get_original_malloc()
{
    return malloc;
}

__aligned_alloc_func_type
__get_original_aligned_alloc()
{
    return _aligned_malloc;
}

__realloc_func_type
__get_original_realloc()
{
    return __original_realloc;
}

std::size_t
__get_page_size()
{
    SYSTEM_INFO __si;
    GetSystemInfo(&__si);
    return __si.dwPageSize;
}

static bool
__do_functions_replacement()
{
    DetourRestoreAfterWith();

    LONG ret = DetourTransactionBegin();
    if (NO_ERROR != ret)
    {
        fprintf(stderr, "Failed function replacement: DetourTransactionBegin returns %ld\n", ret);
        return false;
    }
    ret = DetourUpdateThread(GetCurrentThread());
    if (NO_ERROR != ret)
    {
        fprintf(stderr, "Failed function replacement: DetourUpdateThread returns %ld\n", ret);
        return false;
    }

    // TODO: rarely-used _aligned_offset_* functions are not supported yet
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmicrosoft-cast"
    ret = DetourAttach(&(PVOID&)__original_free, __internal_free);
    if (NO_ERROR != ret)
    {
        fprintf(stderr, "Failed function replacement: free replacement failed with %ld\n", ret);
        return false;
    }
#if _DEBUG
    // _free_dbg is called by delete in debug mode
    ret = DetourAttach(&(PVOID&)__original_free_dbg, __internal_free_dbg);
    if (NO_ERROR != ret)
    {
        fprintf(stderr, "Failed function replacement: free_dbg replacement failed with %ld\n", ret);
        return false;
    }
#endif
    ret = DetourAttach(&(PVOID&)__original_realloc, __internal_realloc);
    if (NO_ERROR != ret)
    {
        fprintf(stderr, "Failed function replacement: realloc replacement failed with %ld\n", ret);
        return false;
    }
    ret = DetourAttach(&(PVOID&)__original_aligned_free, __internal_aligned_free);
    if (NO_ERROR != ret)
    {
        fprintf(stderr, "Failed function replacement: aligned_free replacement failed with %ld\n", ret);
        return false;
    }
    ret = DetourAttach(&(PVOID&)__original_msize, __internal_msize);
    if (NO_ERROR != ret)
    {
        fprintf(stderr, "Failed function replacement: msize replacement failed with %ld\n", ret);
        return false;
    }
    ret = DetourAttach(&(PVOID&)__original_aligned_msize, __internal_aligned_msize);
    if (NO_ERROR != ret)
    {
        fprintf(stderr, "Failed function replacement: _aligned_msize replacement failed with %ld\n", ret);
        return false;
    }
    ret = DetourAttach(&(PVOID&)__original_aligned_realloc, __internal_aligned_realloc);
    if (NO_ERROR != ret)
    {
        fprintf(stderr, "Failed function replacement: _aligned_msize replacement failed with %ld\n", ret);
        return false;
    }
#pragma GCC diagnostic pop

    ret = DetourTransactionCommit();
    if (NO_ERROR != ret)
    {
        fprintf(stderr, "Failed function replacement: DetourTransactionCommit returns %ld\n", ret);
        return false;
    }
    return true;
}

#endif // _WIN64

} // namespace __pstl_offload

#if __linux__
extern "C"
{

#define _PSTL_OFFLOAD_EXPORT __attribute__((visibility("default")))

_PSTL_OFFLOAD_EXPORT void free(void* __ptr)
{
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void __libc_free(void *__ptr)
{
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void* realloc(void* __ptr, std::size_t __new_size)
{
    return ::__pstl_offload::__internal_realloc(__ptr, __new_size);
}

_PSTL_OFFLOAD_EXPORT void* __libc_realloc(void* __ptr, std::size_t __new_size)
{
    return ::__pstl_offload::__internal_realloc(__ptr, __new_size);
}

_PSTL_OFFLOAD_EXPORT std::size_t malloc_usable_size(void* __ptr) noexcept
{
    return ::__pstl_offload::__internal_msize(__ptr);
}

} // extern "C"

_PSTL_OFFLOAD_EXPORT void
operator delete(void* __ptr) noexcept
{
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void
operator delete[](void* __ptr) noexcept
{
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void
operator delete(void* __ptr, std::size_t) noexcept
{
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void
operator delete[](void* __ptr, std::size_t) noexcept
{
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void
operator delete(void* __ptr, std::size_t, std::align_val_t) noexcept
{
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void
operator delete[](void* __ptr, std::size_t, std::align_val_t) noexcept
{
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void
operator delete(void* __ptr, const std::nothrow_t&) noexcept
{
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void
operator delete[](void* __ptr, const std::nothrow_t&) noexcept
{
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void
operator delete(void* __ptr, std::align_val_t) noexcept
{
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void
operator delete[](void* __ptr, std::align_val_t) noexcept
{
    ::__pstl_offload::__internal_free(__ptr);
}

#elif _WIN64

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
extern "C" BOOL WINAPI DllMain( HINSTANCE hInst, DWORD callReason, LPVOID reserved )
{
    BOOL ret = TRUE;

    if ( callReason==DLL_PROCESS_ATTACH && reserved && hInst )
    {
        ret = __pstl_offload::__do_functions_replacement()? TRUE : FALSE;
    }

    return ret;
}
#pragma GCC diagnostic pop

#endif // _WIN64
