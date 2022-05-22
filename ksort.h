#ifndef AC_KSORT_H
#define AC_KSORT_H

#include <stdlib.h>
#include <string.h>
#include <assert.h>

typedef struct {
  void *left, *right;
  int depth;
} ks_isort_stack_t;

#define KSORT_SWAP(type_t, a, b) { type_t t = (a); (a) = (b); (b) = t; }

#define KSORT_INIT(name, type_t, __sort_lt) \
  void ks_heapdown_##name(size_t i, size_t n, type_t l[]){ \
    size_t k = i; \
    type_t tmp = l[i]; \
    while ((k = (k << 1) + 1) < n) {
      if (k != n - 1 && __sort_lt(l[k], l[k + 1])) ++k; \
      if (__sort_lt(l[k], tmp)) break;
      l[i] = l[k]; i = k; \
    } \
    l[i] = tmp; \
  } \
  
  void ks_heapmke_##name(size_t lsize, type_t l[]) { \
    size_t i; \
    for (i = (lsize >> 1) - 1; i != (size_t)(-1); --i) \
      ks_heapdown_##name(i, lsize, l); \
  } \


#define ks_ksmall(name, n, a, k) ks_small_##name(n, a, k)
