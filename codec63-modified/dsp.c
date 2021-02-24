#include <inttypes.h>
#include <math.h>
#include <stdlib.h>

#include "dsp.h"
#include "tables.h"

static void transpose_block(float *in_data, float *out_data)
{
  int i, j;

  for (i = 0; i < 8; ++i)
  {
    for (j = 0; j < 8; ++j)
    {
      out_data[i*8+j] = in_data[j*8+i];
    }
  }
}

static void dct_1d(float *in_data, float *out_data)
{
  int i;


  // varriables
  float32x4_t in, in2;          // in, in2 is varribles used to load in_Data
  float32x4_t r, r2;            // r, r2 is used as temporary varriables to store results.
  
  // load data
  in = vld1q_f32 (in_data);     // load first 4 elements from in_data
  in2 = vld1q_f32 (in_data +4); // load last 4 elements from in_data
  


  for (i = 0; i < 8; ++i)
  {
    // load lookup table into neon varriables
    float32x4_t lookup = vld1q_f32 ((dctlookup_T+i));     // dctlookup_T is a transposed version of dctlookup
    float32x4_t lookup2 = vld1q_f32 ((dctlookup_T[i])+4);
    
    // multiply first 4 elements with with first 4 elements of the lookuptable
    r = vmulq_f32(in, lookup);
    // multiply last 4 elements with with first 4 elements of the lookuptable
    r2 = vmulq_f32(in2, lookup2);
    // add up the 4 elements from r and r2
    r = vaddq_f32(r, r2);
    
    // add vector wide sum to out_data
    out_data[i] = vaddvq_f32(r);
  }
  
  
}

static void idct_1d(float *in_data, float *out_data)
{
  int i;
  
  
  // varriables
  float32x4_t in, in2;          // in, in2 is varribles used to load in_Data
  float32x4_t r, r2;            // r, r2 is used as temporary varriables to store results.
    
  // load data
  in = vld1q_f32 (in_data);     // load first 4 elements from in_data
  in2 = vld1q_f32 (in_data +4); // load last 4 elements from in_data
  
  
  for (i = 0; i < 8; ++i)
  {
    // load lookup table into neon varriables
    float32x4_t lookup = vld1q_f32 ((dctlookup+i));
    float32x4_t lookup2 = vld1q_f32 ((dctlookup[i])+4);

    // multiply first 4 elements with with first 4 elements of the lookuptable
    r = vmulq_f32(in, lookup);
    // multiply last 4 elements with with first 4 elements of the lookuptable
    r2 = vmulq_f32(in2, lookup2);
    // add up the 4 elements from r and r2
    r = vaddq_f32(r, r2);
    
    // add vector wide sum to out_data
    out_data[i] = vaddvq_f32(r);
    
    }
}

static void scale_block(float *in_data, float *out_data)
{
  int u, v;

  for (v = 0; v < 8; ++v)
  {
    for (u = 0; u < 8; ++u)
    {
      float a1 = !u ? ISQRT2 : 1.0f;
      float a2 = !v ? ISQRT2 : 1.0f;

      /* Scale according to normalizing function */
      out_data[v*8+u] = in_data[v*8+u] * a1 * a2;
    }
  }
}

static void quantize_block(float *in_data, float *out_data, uint8_t *quant_tbl)
{
  int zigzag;

  for (zigzag = 0; zigzag < 64; ++zigzag)
  {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    float dct = in_data[v*8+u];

    /* Zig-zag and quantize */
    out_data[zigzag] = (float) round((dct / 4.0) / quant_tbl[zigzag]);
  }
}

static void dequantize_block(float *in_data, float *out_data,
    uint8_t *quant_tbl)
{
  int zigzag;

  for (zigzag = 0; zigzag < 64; ++zigzag)
  {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    float dct = in_data[zigzag];

    /* Zig-zag and de-quantize */
    out_data[v*8+u] = (float) round((dct * quant_tbl[zigzag]) / 4.0);
  }
}

void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl)
{
  float mb[8*8] __attribute((aligned(16)));
  float mb2[8*8] __attribute((aligned(16)));

  int i, v;

  for (i = 0; i < 64; ++i) { mb2[i] = in_data[i]; }

  /* Two 1D DCT operations with transpose */
  for (v = 0; v < 8; ++v) { dct_1d(mb2+v*8, mb+v*8); }
  transpose_block(mb, mb2);
  for (v = 0; v < 8; ++v) { dct_1d(mb2+v*8, mb+v*8); }
  transpose_block(mb, mb2);

  scale_block(mb2, mb);
  quantize_block(mb, mb2, quant_tbl);

  for (i = 0; i < 64; ++i) { out_data[i] = mb2[i]; }
}

void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl)
{
  float mb[8*8] __attribute((aligned(16)));
  float mb2[8*8] __attribute((aligned(16)));

  int i, v;

  for (i = 0; i < 64; ++i) { mb[i] = in_data[i]; }

  dequantize_block(mb, mb2, quant_tbl);
  scale_block(mb2, mb);

  /* Two 1D inverse DCT operations with transpose */
  for (v = 0; v < 8; ++v) { idct_1d(mb+v*8, mb2+v*8); }
  transpose_block(mb2, mb);
  for (v = 0; v < 8; ++v) { idct_1d(mb+v*8, mb2+v*8); }
  transpose_block(mb2, mb);

  for (i = 0; i < 64; ++i) { out_data[i] = mb[i]; }
}

void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
{
  

  // Neon Intrinsics varriables
  uint8x8_t b_1, b_2; 
  uint16x8_t diff, diff2;
  *result = 0; 

  /*  unrolled loop with Neon Intrinsics*/
    b_1 = vld1_u8(block1 + 0*stride); // load 8 elems from block1
    b_2 = vld1_u8(block2 + 0*stride); // load 8 elems from block2
    
    diff = vabdl_u8(b_2, b_1);        // calculate abs difference long, uint8x8_t -> uint16x8_t
    diff2 = vaddq_u16(diff, diff2);
   
    b_1 = vld1_u8(block1 + 1*stride); // load 8 elems from block1
    b_2 = vld1_u8(block2 + 1*stride); // load 8 elems from block2
    
    diff = vabdl_u8(b_2, b_1);        // calculate abs difference long, uint8x8_t -> uint16x8_t
    diff2 = vaddq_u16(diff, diff2);

    b_1 = vld1_u8(block1 + 2*stride); // load 8 elems from block1
    b_2 = vld1_u8(block2 + 2*stride); // load 8 elems from block2
    
    diff = vabdl_u8(b_2, b_1);        // calculate abs difference long, uint8x8_t -> uint16x8_t
    diff2 = vaddq_u16(diff, diff2);
    
    b_1 = vld1_u8(block1 + 3*stride); // load 8 elems from block1
    b_2 = vld1_u8(block2 + 3*stride); // load 8 elems from block2
    
    diff = vabdl_u8(b_2, b_1);        // calculate abs difference long, uint8x8_t -> uint16x8_t
    diff2 = vaddq_u16(diff, diff2);
    
    b_1 = vld1_u8(block1 + 4*stride); // load 8 elems from block1
    b_2 = vld1_u8(block2 + 4*stride); // load 8 elems from block2
    
    diff = vabdl_u8(b_2, b_1);        // calculate abs difference long, uint8x8_t -> uint16x8_t
    diff2 = vaddq_u16(diff, diff2);
 
    b_1 = vld1_u8(block1 + 5*stride); // load 8 elems from block1
    b_2 = vld1_u8(block2 + 5*stride); // load 8 elems from block2
    
    diff = vabdl_u8(b_2, b_1);        // calculate abs difference long, uint8x8_t -> uint16x8_t
    diff2 = vaddq_u16(diff, diff2);
  
    b_1 = vld1_u8(block1 + 6*stride); // load 8 elems from block1
    b_2 = vld1_u8(block2 + 6*stride); // load 8 elems from block2
    
    diff = vabdl_u8(b_2, b_1);        // calculate abs difference long, uint8x8_t -> uint16x8_t
    diff2 = vaddq_u16(diff, diff2);
 
    b_1 = vld1_u8(block1 + 7*stride); // load 8 elems from block1
    b_2 = vld1_u8(block2 + 7*stride); // load 8 elems from block2
    
    diff = vabdl_u8(b_2, b_1);        // calculate abs difference long, uint8x8_t -> uint16x8_t
    diff2 = vaddq_u16(diff, diff2);
    *result += vaddvq_u16(diff2);      // vector wide sum
    

  /*
  for (v = 0; v < 8; ++v)
  {
    
    // Neon Intrinsics
    b_1 = vld1_u8(block1 + v*stride); // load 8 elems from block1
    b_2 = vld1_u8(block2 + v*stride); // load 8 elems from block2
    
    diff = vabdl_u8(b_2, b_1);        // calculate abs difference long, uint8x8_t -> uint16x8_t
    *result += vaddvq_u16(diff);      // vector wide sum
    



    //  inline-assembly. 
    uint8_t *blk_1;
		uint8_t *blk_2;
		
		blk_1 = block1 + v*stride;
		blk_2 = block2 + v*stride;
    
		  __asm__ (
		    "ld1 {v0.8h}, [%0]\n\t"         // load result value
			  "ld1 {v1.8b}, [%1]\n\t"         // load block1
			  "ld1 {v2.8b}, [%2]\n\t"         // load block2
			  
			  "uabdl v3.8h, v1.8b, v2.8b\n\t"  // calculate absolute difference long
			  "addv h3, v3.8h\n\t"            // vector wide sum store in register v3
			  
			  "add v0.8h, v0.8h, v3.8h\n\t"   // add v3 to result value
			  "st1 {v0.8h}, [%0]\n\t"         // store result value 
		  : // Output operands
		  : "r" (result), "r" (blk_1), "r" (blk_2) // Input operands
		  : "v0", "v1", "v2", "v3", "memory"  // dirty registers
		  );*/
  //}
}
