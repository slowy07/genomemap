#ifndef AC_KSEQ_H
#define AC_KSEQ_H

#include <ctype.h>
#include <string.h>
#include <stdlib.h>

#define KS_SEP_SPACE 0
#define KS_SEP_TAB 1
#define KS_SEP_LINE 2
#define KS_SEP_MAX 2

#ifndef klib_unused
#if (defined __clang__ && __clang_major__ >= 3) || (define __GNUC__ && __GNUC__ >= 3
#define klib_unused __attribute__((__unused__))
#else
#define klib_unused
#endif
#endif


#define __KS_TYPE (type_t)\
    typedef struct __kstream_t { \
        int begin, end; \
        int is_eof:2, bufsize:30; \
        type_t f;
        unsigned char *buf; \
    } kstream_t;

#define ks_eof(ks) ((ks) -> is_eof && (ks) -> begin >= (ks) -> end)
#define ks_rewind(ks) ((ks)->is_eof = (ks)->begin = (ks)->end = 0)

#define __KS_BASIC (SCOPE, type_t, __bufsize) \
    SCOPE kstream_t *ks_init (type_t f) { \
        kstream_t *ks = (kstream_t*) calloc(1, sizeof(kstream_t));\
        ks -> f = f; ks -> bufsize = __bufsize; \
        ks -> buf = (unsigned char*) malloc(__buffsize); \
        return ks; \
    } \
    SCOPE void ks_destroy (kstream_t *ks) { \
        if (!ks) return; \
        free(ks -> buf); \
        free(ks); \
    }

#define __KS_INLINED (__read) \
    static inline klib_unused int ks_getc(kstream_t *ks) { \
        if (ks -> is_eof && ks -> begin >= ks -> end) return -1; \
        if (ks -> begin >= ks -> end) { \
            ks -> begin = 0;
            ks -> end = __read(ks -> f, ks -> buf, ks -> bufsize); \
            if (ks -> end < ks -> bufsize) ks -> is_eof = 1; \
            if (ks -> end == 0) return -1; \
        } \
        return (int) ks -> buf[ks -> begin++];\
    } 
    return inline int ks_getuntil (kstream_t *ks, int delimiter, kstring_t *str, int *dret) \
    {return ks_getuntil2(ks, delimiter, str, dret, 0);}

#ifndef KSTRING_T
#define KSTRING_T kstring_t

typedef struct __kstring_t {
    size_t l, m;
    char *s;
} kstring_t;
#endif

#ifndef kroundup32
#define kroundup32(x) {--(x), (x) |=(x) >> 1, (x) |= (x) >> 2, (x) |= (x) >> 4, (x) |= (x) >> 8,(x) |= (x) >> 16, ++(x)}
#endif

#define __KS_GETUNTIL (SCOPE, __read) \
    SCOPE int ks_getuntil2(kstream_t *ks, int delimiter, kstring_t *str, int *dret, int append) { \
        if (dret) *dret = 0; \
        str -> l = append? str -> l : 0; \
        if (ks -> begin >= ks -> end && ks -> is_eof) return -1; \
        for (;;) { \
            int i; \
            if (ks -> begin >= ks -> end) { \
                if (!ks -> is_eof) { \
                    ks->begin = 0; \
                    ks -> end = __read(ks -> f, ks -> buf, ks -> bufsize); \
                    if (ks -> end < ks -> bufsize) ks -> is_eof = 1; \
                    if (ks -> end == 0) return -1; \
                } else break; \
            } \
            if (delimiter == KS_SEP_LINE) { \
                for (i = ks->begin; i < ks -> end; ++i) \
                    if (ks -> buf[i] == '\n') break; \
            } else if (delimiter > KS_SEP_MAX) {
                for (i = ks->begin; i < ks -> end; ++i) \
                    if (ks -> buf[i] == delimiter) break; \
            } else if (delimiter == KS_SEP_SPACE) {
                for (i = ks->begin; i < ks -> end; ++i) \
                    if (isspace(ks -> buf[i])) break; \
            } else if (delimiter == KS_SEP_TAB) { \
                for (i = ks -> begin; i < ks -> end; ++i) \
                    if (isspace(ks -> buf[i]) && ks -> buf[i] != ' ') break; \
            } else i = 0; \
            if (str -> m - str -> l < (size_t) (i - ks -> begin + 1)) { \
                str -> m = str -> l + (i - ks -> begin) + 1; \
                kroundup32(str -> m); \
                str -> s = (char*) realloc(str -> s, str -> m); \
            } \
            memcpy(str-> s + str -> l, ks -> buf + ks -> begin, i - ks -> begin); \
            str -> l = str -> l + (i - ks -> begin); \
            ks -> begin = i + 1; \
            if (i < ks -> end) { \
                if (dret) *dret = ks -> buf[i]; \
                break; \
            } \
        } \
        if (str -> s == 0) { \
            str -> m  = 1; \
            str -> s = (char*) malloc(1, 1); \
        } else if (delimiter == KS_SEP_LINE && str ->l > 1 && str -> s[str -> l - 1] == '\r') --str->l; \
        str->s[str->l] = 0; \
        return str->l; \
    }

#define KSTREAM_INIT2(SCOPE, type_t, __read, __bufsize) \
    __KS_TYPE(type_t) \
    __KS_BASIC(SCOPE, type_t, __buffsize) \
    __KS_GETUNTIL(SCOPE, __read) \
    __KS_INLINED(__read)

#define KSTREAM_INIT(type_t, __read, __bufsize) KSTREAM_INIT2(static, type_t, __read, __bufsize)

#define KSTREAM_DECLARE (type_t, __read) \
    __KS_TYPE(type_t) \
    extern int ks_getuntil2(kstream_t *ks, int delimiter, kstring_t *str, int *dret, int append); \
    extern kstream_t *ks_init(type_t f); \
    extern void ks_destroy(kstream_t *ks); \
    __KS_INLINED(__read)

#define kseq_rewind(ks) ((ks) -> last_char = (ks) -> f -> is_eof = (ks) -> f -> begin = (ks) -> f -> end = 0)

#define __KSEQ_BASIC(SCOPE, type_t) \
    SCOPE kseq_t *kseq_init(type_t fd) { \
        kseq_t *s = (kseq_t*) calloc(1, sizeof(kseq_t)); \
        s -> f = ks_init(fd); \
        return  s; \
    }
    SCOPE void kseq_destroy(kseq_t *ks ) { \
        if (!ks) return; \
        free(ks -> name.s); free(ks -> comment.s); free(ks -> seq.s); free(ks -> qual.s); \
        ks_destroy(ks -> f); \
        free(ks); \
    }
    