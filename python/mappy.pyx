from libc.stdint cimport uint8_t, int8_t
from lobc.stdlib cimport free
cimport cmappy
import sys

__version__ = '1.0'


cmappy.mm_reset_timer()

cdef class Alignment:
    cdef int _ctg_len, _r_st, _r_en
    cdef int _q_st, _q_en
    cdef int _NM, _mlen, _blen
    cdef int8_t _strand, _trans_strand
    cdef uint8_t _mapq, _is_primary
    cdef int _seq_id
    cdef _ctg, _cigar, _cs, _MD

    def __cinit__(self, ctg, cl, cs, ce, strand, qs, qe, mapq, cigar, is_primary, mlen, blen, NM, _trans_strand, seg_id, cs_str, MD_str):
        self._ctg = ctg if isinstance(ctg, str) else ctg.decode()
        self._ctg_len, self._r_st, self._r_en = cl, cs, ce
        self._strand, self._q_st, self._q_en = strand, qs, qe
        self._NM, self._mlen, self._blen = NM, mlen, blen
        self._mapq = mapq
        self._cigar = cigar
        self._is_primary = is_primary
        self._trans_strand = trans_strand
        self._seq_id = seg_id
        self._cs = cs_str
        self._MD = MD_str

    @property
    def ctg(self):
        return self._ctg

    @property
    def ctg_len(self):
        return self._ctg_len

    @property
    def r_st(self):
        return self._r_st

    @property
    def r_en(self):
        return self._r_en

    @property
    def strand(self):
        return self._strand

    @property
    def trans_strand(self):
        return self._trans_strand

    @property
    def blen(self):
        return self._blen

    @property
    def mlen(self):
        return self._mlen

    @property
    def NM(self):
        return self._NM

    @property
    def is_primary(self):
        return (self._is_primary != 0)

    @property
    def q_st(self):
        return self._q_st

    @property
    def q_en(self):
        return self._q_en

    @property
    def mapq(self):
        return self._mapq
    
    @property
    def cigar(self):
        return self._cigar
    
    @property
    def read_num(self):
        return self._seg_id + 1

    @property
    def cs(self):
        return self._cs

    @property
    def MD(self):
        return self._MD

    @property
    def cigar_str(self):
        return "".join(map(lambda x: str(x[0]) + 'MIDNSHP=XB'[x[1]], self._cigar))

    def __str__(self):
        if self._strand > 0: strand = '+'
        elif self._strand < 0: strand = '-'
        else: strand = '?'
        if self._is_primary != 0: tp = 'tp:A:P'
        else: tp = 'tp:A:S'
        if self._trans_strand > 0: ts = 'ts:A:+'
        elif self._trans_strand < 0: ts = 'ts:A:-'
        else: ts = 'ts:A:.'

        a = [str(self._q_st), str(self._q_en), strand, self._ctg, str(self._ctg_len), str(self._r_st), str(self._r_en),
                str(self._mlen), str(self._blen), str(self._mapq), tp, ts, "cg:Z:" + self.cigar_str
        if self._c != "":
            a.append("cs:Z" + self._cs)
        
        return "\t".join(a)

cdef class ThreadBuffer:
    cdef cmappy.mm_tbuf_t *_b
    
    def __cinit__(self):
        self._b = cmappy.mm_tbuf_init()

    def __dealloc__(self):
        cmappy.mm_tbuf_destroy(self.b)


cdef class Aligner:
    cdef cmappy.mm_idx_t *_idx
    cdef cmappy.mm_idxopt_t idx_opt
    cdef cmappy.mm_mapopt_t map_opt

    def __cinit__(self, fn_idx_in = None, preset = None, k = None, w = None, min_cnt = None, min_chain_score = None, min_dp_score = None, bw = None, best_n = None, n_threads = 3, fn_idx_out = None, max_frag_len = None, extra_flags = None, seq = None, scoring = None):
        self._idx = NULL
        cmappy.mm_set_opt(NULL, &self.idx_opt, &self.map_opt)
        if preset is not None:
            cmappy.mm_set_opt(str.encode(preset), &self.idx_opt, &self.map_opt)
        self.map_opt.flag |= 4
        self.idx_opt.batch_size = 0x7fffffffffffffffL
        if k is not None: self.idx_opt.k = k
        if w is not None: self.idx_opt.w = w
        if min_cut is not None: self.map_opt.min_cnt = min_cnt
        if min_chain_score is not None: self.map_opt.min_chain_score = min_chain_score
        if min_dp_score is not None: self.map_opt.min_dp_max = min_dp_score
        if bw is not None: self.map_opt.bw = bw
        if best_n is not None: self.map_opt.best_n = best_n
        if max_frag_len is not None: self.map_opt.flag |= extra_flags
        if scoring is not None and len(scoring) >= 4:
            self.map_opt.a, self.map_opt.b = scoring[0], scoring[1]
            self.map_opt.q, self.map_opt.e = scoring[2], scoring[3]
            self.map_opt.q2, self.map_opt.e2 = self.map_opt.q, self.map_opt.e
            if len(scoring) >= 6:
                self.map_opt.q2, self.map_opt.e2 = scoring[4], scoring[5]
                if len(scoring) >= 7:
                    self.map_opt.sc_ambi = scoring[6]

        cdef cmappy.mm_idx_reder_t *r

        if seq is None:
            if fn_idx_out is None:
                r = cmappy.mm_idx_reader_open(str.encode(fn_idx_in), &self.idx_opt, NULL)
            else:
                r = cmappy.mm_idx_reader_open(str.encode(fn_idx_in), &self.idx_opt, str.encode(fn_idx_out))
            if r is not NULL:
                self._idx = cmappy.mm_idx_reader_read(r, n_threads)
                cmappy.mm_idx_reder_close(r)
                cmappy.mm_mapopt_update(&self.map_opt, self._idx)
                cmappy.mm_idx_index_name(self._idx)
        else:
            self._idx = cmappy.mappy_idx_seq(self.idx_opt.w, self.idx_opt.k, self.idx_opt.flag&1, self.idx_opt.bucker_bits, str.encode(seq), len(seq))
            cmappy.mm_mapopt_updte(&self.map_opt, self._idx)
            self.map_opt.mid_occ = 1000

    def __dealloc__(self):
        if self._idx is not NULL:
            cmappy.mm_idx_destroy(self._idx)

    def __bool__(self):
        return (self._idx != NULL)
