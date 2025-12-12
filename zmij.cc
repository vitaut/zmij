// A double-to-string conversion algorithm based on Schubfach.
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE).

#if __has_include("zmij.h")
#  include "zmij.h"
#endif

#include <assert.h>  // assert
#include <stdint.h>  // uint64_t
#include <string.h>  // memcpy

#include <limits>  // std::numeric_limits

namespace {

struct uint128 {
  uint64_t hi;
  uint64_t lo;

  [[maybe_unused]] explicit operator uint64_t() const noexcept { return lo; }

  [[maybe_unused]] auto operator>>(int shift) const noexcept -> uint128 {
    assert(shift >= 64 && shift < 128);
    return {0, hi >> (shift - 64)};
  }
};

#ifdef __SIZEOF_INT128__
using uint128_t = unsigned __int128;
#else
using uint128_t = uint128;
#endif  // __SIZEOF_INT128__

// 126-bit significands of overestimates of powers of 10.
// Generated with gen-pow10.py.
const uint128 pow10_significands[] = {
    {0x7fbbd8fe5f5e6e27, 0x92f4744e09dd87be},  // -292
    {0x4fd5679efb9b04d8, 0xbbd8c8b0c62a74d8},  // -291
    {0x63cac186ba81c60e, 0xeacefadcf7b5120c},  // -290
    {0x7cbd71e869223792, 0xa582b99435a25690},  // -289
    {0x4df6673141b562bb, 0xa771b3fca185761a},  // -288
    {0x617400fd9222bb6a, 0x914e20fbc9e6d3a0},  // -287
    {0x79d1013cf6ab6a45, 0x35a1a93abc608888},  // -286
    {0x4c22a0c61a2b226b, 0x418509c4b5bc5556},  // -285
    {0x5f2b48f7a0b5eb06, 0x11e64c35e32b6aaa},  // -284
    {0x76f61b3588e365c7, 0x965fdf435bf64556},  // -283
    {0x4a59d101758e1f9c, 0xbdfbeb8a1979eb56},  // -282
    {0x5cf04541d2f1a783, 0xed7ae66c9fd8662a},  // -281
    {0x742c569247ae1164, 0xe8d9a007c7ce7fb6},  // -280
    {0x489bb61b6ccccadf, 0x11880404dce10fd2},  // -279
    {0x5ac2a3a247fffd96, 0xd5ea0506141953c6},  // -278
    {0x71734c8ad9fffcfc, 0x8b648647991fa8b8},  // -277
    {0x46e80fd6c83ffe1d, 0xd71ed3ecbfb3c972},  // -276
    {0x58a213cc7a4ffda5, 0x4ce688e7efa0bbd0},  // -275
    {0x6eca98bf98e3fd0e, 0xa0202b21eb88eac2},  // -274
    {0x453e9f77bf8e7e29, 0x24141af5333592ba},  // -273
    {0x568e4755af721db3, 0x6d1921b28002f768},  // -272
    {0x6c31d92b1b4ea520, 0x485f6a1f2003b542},  // -271
    {0x439f27baf1112734, 0x2d3ba2537402514a},  // -270
    {0x5486f1a9ad557101, 0x388a8ae85102e59c},  // -269
    {0x69a8ae1418aacd41, 0x86ad2da265439f02},  // -268
    {0x42096ccc8f6ac048, 0xf42c3c857f4a4362},  // -267
    {0x528bc7ffb345705b, 0x31374ba6df1cd43a},  // -266
    {0x672eb9ffa016cc71, 0xfd851e9096e40948},  // -265
    {0x407d343fc40e3fc7, 0x3e73331a5e4e85ce},  // -264
    {0x509c814fb511cfb9, 0x0e0fffe0f5e22742},  // -263
    {0x64c3a1a3a25643a7, 0x5193ffd9335ab112},  // -262
    {0x7df48a0c8aebd491, 0x25f8ffcf80315d56},  // -261
    {0x4eb8d647d6d364da, 0xb7bb9fe1b01eda56},  // -260
    {0x62670bd9cc883e11, 0x65aa87da1c2690ea},  // -259
    {0x7b00ced03faa4d95, 0xbf1529d0a3303526},  // -258
    {0x4ce0814227ca707d, 0x976d3a2265fe2138},  // -257
    {0x6018a192b1bd0c9c, 0xfd4888aaff7da986},  // -256
    {0x781ec9f75e2c4fc4, 0x3c9aaad5bf5d13e6},  // -255
    {0x4b133e3a9adbb1da, 0xa5e0aac5979a2c70},  // -254
    {0x5dd80dc941929e51, 0x4f58d576fd80b78c},  // -253
    {0x754e113b91f745e5, 0xa32f0ad4bce0e570},  // -252
    {0x4950cac53b3a8baf, 0x85fd66c4f60c8f66},  // -251
    {0x5ba4fd768a092e9b, 0x677cc076338fb33e},  // -250
    {0x728e3cd42c8b7a42, 0x415bf093c073a00e},  // -249
    {0x4798e6049bd72c69, 0x68d9765c5848440a},  // -248
    {0x597f1f85c2ccf783, 0xc30fd3f36e5a550c},  // -247
    {0x6fdee76733803564, 0xb3d3c8f049f0ea4e},  // -246
    {0x45eb50a08030215e, 0xf0645d962e369272},  // -245
    {0x576624c8a03c29b6, 0xac7d74fbb9c4370e},  // -244
    {0x6d3fadfac84b3424, 0x579cd23aa83544d0},  // -243
    {0x4447ccbcbd2f0096, 0xb6c20364a9214b02},  // -242
    {0x5559bfebec7ac0bc, 0x6472843dd3699dc2},  // -241
    {0x6ab02fe6e79970eb, 0x7d8f254d48440534},  // -240
    {0x42ae1df050bfe693, 0x2e7977504d2a8340},  // -239
    {0x5359a56c64efe037, 0xfa17d52460752410},  // -238
    {0x68300ec77e2bd845, 0xf89dca6d78926d14},  // -237
    {0x411e093caedb672b, 0xbb629e846b5b842e},  // -236
    {0x51658b8bda9240f6, 0xaa3b462586326538},  // -235
    {0x65beee6ed136d134, 0x54ca17aee7befe86},  // -234
    {0x7f2eaa0a85848581, 0x69fc9d9aa1aebe28},  // -233
    {0x4f7d2a469372d370, 0xe23de280a50d36d8},  // -232
    {0x635c74d8384f884d, 0x1acd5b20ce50848e},  // -231
    {0x7c33920e46636a60, 0x6180b1e901e4a5b2},  // -230
    {0x4da03b48ebfe227c, 0x3cf06f31a12ee790},  // -229
    {0x61084a1b26fdab1b, 0x4c2c8afe097aa174},  // -228
    {0x794a5ca1f0bd15e2, 0x1f37adbd8bd949d0},  // -227
    {0x4bce79e536762dad, 0x5382cc967767ce22},  // -226
    {0x5ec2185e8413b918, 0xa8637fbc1541c1aa},  // -225
    {0x76729e762518a75e, 0xd27c5fab1a923216},  // -224
    {0x4a07a309d72f689b, 0x438dbbcaf09b5f4e},  // -223
    {0x5c898bcc4cfb42c2, 0x14712abdacc23720},  // -222
    {0x73abeebf603a1372, 0x998d756d17f2c4e8},  // -221
    {0x484b75379c244c27, 0x9ff869642ef7bb12},  // -220
    {0x5a5e5285832d5f31, 0x87f683bd3ab5a9d6},  // -219
    {0x70f5e726e3f8b6fd, 0xe9f424ac8963144c},  // -218
    {0x4699b0784e7b725e, 0xb23896ebd5ddecb0},  // -217
    {0x58401c96621a4ef6, 0x5ec6bca6cb5567da},  // -216
    {0x6e5023bbfaa0e2b3, 0xf6786bd07e2ac1d2},  // -215
    {0x44f216557ca48db0, 0x7a0b43624edab924},  // -214
    {0x562e9beadbcdb11c, 0x988e143ae291676c},  // -213
    {0x6bba42e592c11d63, 0xbeb199499b35c146},  // -212
    {0x435469cf7bb8b25e, 0x572effce010198cc},  // -211
    {0x542984435aa6def5, 0xecfabfc18141ff00},  // -210
    {0x6933e554315096b3, 0x68396fb1e1927ebe},  // -209
    {0x41c06f549ed25e30, 0x2123e5cf2cfb8f38},  // -208
    {0x52308b29c686f5bc, 0x296cdf42f83a7306},  // -207
    {0x66bcadf43828b32b, 0x33c81713b6490fc6},  // -206
    {0x4035ecb8a3196ffb, 0x005d0e6c51eda9dc},  // -205
    {0x504367e6cbdfcbf9, 0xc074520766691454},  // -204
    {0x645441e07ed7bef8, 0x3091668940035968},  // -203
    {0x7d6952589e8daeb6, 0x3cb5c02b90042fc2},  // -202
    {0x4e61d37763188d31, 0xe5f1981b3a029dda},  // -201
    {0x61fa48553bdeb07e, 0x5f6dfe2208834550},  // -200
    {0x7a78da6a8ad65c9d, 0xf7497daa8aa416a4},  // -199
    {0x4c8b888296c5f9e2, 0xba8dee8a96a68e26},  // -198
    {0x5fae6aa33c77785b, 0x69316a2d3c5031b0},  // -197
    {0x779a054c0b955672, 0x437dc4b88b643e1c},  // -196
    {0x4ac0434f873d5607, 0x6a2e9af3571ea6d2},  // -195
    {0x5d705423690cab89, 0x44ba41b02ce65086},  // -194
    {0x74cc692c434fd66b, 0x95e8d21c381fe4a6},  // -193
    {0x48ffc1bbaa11e603, 0x3db18351a313eee8},  // -192
    {0x5b3fb22a94965f84, 0x0d1de4260bd8eaa2},  // -191
    {0x720f9eb539bbf765, 0x10655d2f8ecf254a},  // -190
    {0x4749c33144157a9f, 0x2a3f5a3db9417750},  // -189
    {0x591c33fd951ad946, 0xf4cf30cd2791d522},  // -188
    {0x6f6340fcfa618f98, 0xb202fd0071764a6c},  // -187
    {0x459e089e1c7cf9bf, 0x6f41de2046e9ee84},  // -186
    {0x57058ac5a39c382f, 0x4b1255a858a46a24},  // -185
    {0x6cc6ed770c83463b, 0x1dd6eb126ecd84ac},  // -184
    {0x43fc546a67d20be4, 0xf2a652eb854072ec},  // -183
    {0x54fb698501c68ede, 0x2f4fe7a666908fa8},  // -182
    {0x6a3a43e642383295, 0xbb23e1900034b390},  // -181
    {0x42646a6fe9631f9d, 0x94f66cfa0020f03a},  // -180
    {0x52fd850be3bbe784, 0xfa34083880292c4a},  // -179
    {0x67bce64edcaae166, 0x38c10a46a033775c},  // -178
    {0x40d60ff149eaccdf, 0xe378a66c24202a9a},  // -177
    {0x510b93ed9c658017, 0xdc56d0072d283540},  // -176
    {0x654e78e9037ee01d, 0xd36c8408f8724290},  // -175
    {0x7ea21723445e9825, 0x4847a50b368ed332},  // -174
    {0x4f254e760abb1f17, 0x4d2cc72702194400},  // -173
    {0x62eea2138d69e6dd, 0x2077f8f0c29f9500},  // -172
    {0x7baa4a9870c46094, 0x6895f72cf3477a40},  // -171
    {0x4d4a6e9f467abc5c, 0xc15dba7c180cac68},  // -170
    {0x609d0a4718196b73, 0xf1b5291b1e0fd782},  // -169
    {0x78c44cd8de1fc650, 0xee227361e593cd62},  // -168
    {0x4b7ab0078ad3dbf2, 0x94d5881d2f7c605e},  // -167
    {0x5e595c096d88d2ef, 0x3a0aea247b5b7874},  // -166
    {0x75efb30bc8eb07ab, 0x088da4ad9a325692},  // -165
    {0x49b5cfe75d92e4ca, 0xe55886ec805f761c},  // -164
    {0x5c2343e134f79dfd, 0x9eaea8a7a07753a2},  // -163
    {0x732c14d98235857d, 0x065a52d18895288a},  // -162
    {0x47fb8d07f161736e, 0x23f873c2f55d3956},  // -161
    {0x59fa7049edb9d049, 0xacf690b3b2b487ac},  // -160
    {0x70790c5c6928445c, 0x183434e09f61a998},  // -159
    {0x464ba7b9c1b92ab9, 0x8f20a10c639d09fe},  // -158
    {0x57de91a832277567, 0xf2e8c94f7c844c7e},  // -157
    {0x6dd636123eb152c1, 0xefa2fba35ba55f9e},  // -156
    {0x44a5e1cb672ed3b9, 0x35c5dd4619475bc2},  // -155
    {0x55cf5a3e40fa88a7, 0x833754979f9932b4},  // -154
    {0x6b4330cdd1392ad1, 0x640529bd877f7f60},  // -153
    {0x4309fe80a2c3bac2, 0xde833a1674afaf9c},  // -152
    {0x53cc7e20cb74a973, 0x9624089c11db9b84},  // -151
    {0x68bf9da8fe51d3d0, 0x7bad0ac316528264},  // -150
    {0x4177c2899ef32462, 0x4d4c26b9edf3917e},  // -149
    {0x51d5b32c06afed7a, 0xe09f3068697075de},  // -148
    {0x664b1ff7085be8d9, 0x98c6fc8283cc9356},  // -147
    {0x7fdde7f4ca72e30f, 0xfef8bba324bfb82a},  // -146
    {0x4feab0f8fe87cde9, 0xff5b7545f6f7d31a},  // -145
    {0x63e55d373e29c164, 0x7f32529774b5c7e2},  // -144
    {0x7cdeb4850db431bd, 0x9efee73d51e339da},  // -143
    {0x4e0b30d328909f16, 0x835f5086532e0428},  // -142
    {0x618dfd07f2b4c6dc, 0x243724a7e7f98532},  // -141
    {0x79f17c49ef61f893, 0x2d44edd1e1f7e67e},  // -140
    {0x4c36edae359d3b5b, 0xfc4b14a32d3af010},  // -139
    {0x5f44a919c3048a32, 0xfb5dd9cbf889ac12},  // -138
    {0x7715d36033c5acbf, 0xba35503ef6ac1718},  // -137
    {0x4a6da41c205b8bf7, 0xd46152275a2b8e70},  // -136
    {0x5d090d2328726ef5, 0xc979a6b130b6720a},  // -135
    {0x744b506bf28f0ab3, 0x3bd8105d7ce40e8c},  // -134
    {0x48af1243779966b0, 0x05670a3a6e0e8918},  // -133
    {0x5adad6d4557fc05c, 0x06c0ccc909922b5e},  // -132
    {0x71918c896adfb073, 0x0870fffb4bf6b636},  // -131
    {0x46faf7d5e2cbce47, 0xe5469ffd0f7a31e2},  // -130
    {0x58b9b5cb5b7ec1d9, 0xde9847fc5358be5a},  // -129
    {0x6ee8233e325e7250, 0x563e59fb682eedf0},  // -128
    {0x45511606df7b0772, 0x35e6f83d211d54b6},  // -127
    {0x56a55b889759c94e, 0xc360b64c6964a9e4},  // -126
    {0x6c4eb26abd303ba2, 0x7438e3df83bdd45c},  // -125
    {0x43b12f82b63e2545, 0x88a38e6bb256a4ba},  // -124
    {0x549d7b6363cdae96, 0xeacc72069eec4de8},  // -123
    {0x69c4da3c3cc11a3c, 0xa57f8e8846a76162},  // -122
    {0x421b0865a5f8b065, 0xe76fb9152c289cde},  // -121
    {0x52a1ca7f0f76dc7f, 0x614ba75a7732c416},  // -120
    {0x674a3d1ed354939f, 0x399e913114ff751a},  // -119
    {0x408e66334414dc43, 0x84031abead1fa930},  // -118
    {0x50b1ffc0151a1354, 0x6503e16e5867937c},  // -117
    {0x64de7fb01a609829, 0x7e44d9c9ee81785c},  // -116
    {0x7e161f9c20f8be33, 0xddd6103c6a21d672},  // -115
    {0x4ecdd3c1949b76e0, 0x6aa5ca25c2552608},  // -114
    {0x628148b1f9c25498, 0x854f3caf32ea6f8a},  // -113
    {0x7b219ade7832e9be, 0xa6a30bdaffa50b6c},  // -112
    {0x4cf500cb0b1fd217, 0x2825e768dfc72724},  // -111
    {0x603240fdcde7c69c, 0xf22f614317b8f0ec},  // -110
    {0x783ed13d4161b844, 0x2ebb3993dda72d28},  // -109
    {0x4b2742c648dd132a, 0x9d3503fc6a887c38},  // -108
    {0x5df11377db1457f5, 0x448244fb852a9b46},  // -107
    {0x756d5855d1d96df2, 0x95a2d63a66754218},  // -106
    {0x49645735a327e4b7, 0x9d85c5e480094950},  // -105
    {0x5bbd6d030bf1dde5, 0x84e7375da00b9ba4},  // -104
    {0x72acc843ceee555e, 0xe6210535080e828c},  // -103
    {0x47abfd2a6154f55b, 0x4fd4a34125091198},  // -102
    {0x5996fc74f9aa32b2, 0x23c9cc116e4b55fe},  // -101
    {0x6ffcbb923814bf5e, 0xacbc3f15c9de2b7c},  // -100
    {0x45fdf53b630cf79b, 0x2bf5a76d9e2adb2e},  //  -99
    {0x577d728a3bd03581, 0xf6f3114905b591fa},  //  -98
    {0x6d5ccf2ccac442e2, 0x74afd59b4722f678},  //  -97
    {0x445a017bfebaa9cd, 0x88ede5810c75da0c},  //  -96
    {0x557081dafe695440, 0xeb295ee14f93508e},  //  -95
    {0x6acca251be03a951, 0x25f3b699a37824b0},  //  -94
    {0x42bfe57316c249d2, 0xb7b85220062b16ee},  //  -93
    {0x536fdecfdc72dc47, 0x65a666a807b5dcaa},  //  -92
    {0x684bd683d38f9359, 0x3f10005209a353d4},  //  -91
    {0x412f66126439bc17, 0xc76a003346061466},  //  -90
    {0x517b3f96fd482b1d, 0xb94480401787997e},  //  -89
    {0x65da0f7cbc9a35e5, 0x2795a0501d697fde},  //  -88
    {0x7f50935bebc0c35e, 0x717b086424c3dfd6},  //  -87
    {0x4f925c1973587a1b, 0x06ece53e96fa6be6},  //  -86
    {0x6376f31fd02e98a1, 0xc8a81e8e3cb906de},  //  -85
    {0x7c54afe7c43a3eca, 0x3ad22631cbe74896},  //  -84
    {0x4db4edf0daa4673e, 0x64c357df1f708d5e},  //  -83
    {0x6122296d114d810d, 0xfdf42dd6e74cb0b6},  //  -82
    {0x796ab3c855a0e151, 0x7d71394ca11fdce2},  //  -81
    {0x4be2b05d35848cd2, 0xee66c3cfe4b3ea0e},  //  -80
    {0x5edb5c7482e5b007, 0xaa0074c3dde0e492},  //  -79
    {0x76923391a39f1c09, 0x948091f4d5591db6},  //  -78
    {0x4a1b603b06437185, 0xfcd05b390557b292},  //  -77
    {0x5ca23849c7d44de7, 0x7c04720746ad9f36},  //  -76
    {0x73cac65c39c96161, 0x5b058e8918590704},  //  -75
    {0x485ebbf9a41ddcdc, 0xd8e37915af37a462},  //  -74
    {0x5a766af80d255414, 0x0f1c575b1b058d7a},  //  -73
    {0x711405b6106ea919, 0x12e36d31e1c6f0da},  //  -72
    {0x46ac8391ca4529af, 0xabce243f2d1c5688},  //  -71
    {0x5857a4763cd6741b, 0x96c1ad4ef8636c2a},  //  -70
    {0x6e6d8d93cc0c1122, 0x7c7218a2b67c4734},  //  -69
    {0x4504787c5f878ab5, 0x8dc74f65b20dac80},  //  -68
    {0x5645969b77696d62, 0xf139233f1e9117a0},  //  -67
    {0x6bd6fc425543c8bb, 0xad876c0ee6355d88},  //  -66
    {0x43665da9754a5d75, 0x4c74a3894fe15a76},  //  -65
    {0x543ff513d29cf4d2, 0x9f91cc6ba3d9b114},  //  -64
    {0x694ff258c7443207, 0x47763f868cd01d58},  //  -63
    {0x41d1f7777c8a9f44, 0x8ca9e7b418021258},  //  -62
    {0x524675555bad4715, 0xafd461a11e0296ec},  //  -61
    {0x66d812aab29898db, 0x1bc97a0965833ca8},  //  -60
    {0x40470baaaf9f5f88, 0xf15dec45df7205ea},  //  -59
    {0x5058ce955b87376b, 0x2db56757574e8764},  //  -58
    {0x646f023ab2690545, 0xf922c12d2d22293c},  //  -57
    {0x7d8ac2c95f034697, 0x776b7178786ab38a},  //  -56
    {0x4e76b9bddb620c1e, 0xaaa326eb4b42b036},  //  -55
    {0x6214682d523a8f26, 0x554bf0a61e135c44},  //  -54
    {0x7a998238a6c932ef, 0xea9eeccfa5983356},  //  -53
    {0x4c9ff163683dbfd5, 0xf2a35401c77f2016},  //  -52
    {0x5fc7edbc424d2fcb, 0x6f4c2902395ee81a},  //  -51
    {0x77b9e92b52e07bbe, 0x4b1f3342c7b6a222},  //  -50
    {0x4ad431bb13cc4d56, 0xeef38009bcd22556},  //  -49
    {0x5d893e29d8bf60ac, 0xaab0600c2c06aeaa},  //  -48
    {0x74eb8db44eef38d7, 0xd55c780f37085a54},  //  -47
    {0x49133890b1558386, 0xe559cb0982653876},  //  -46
    {0x5b5806b4ddaae468, 0x9eb03dcbe2fe8692},  //  -45
    {0x722e086215159d82, 0xc65c4d3edbbe2836},  //  -44
    {0x475cc53d4d2d8271, 0xbbf9b0474956d922},  //  -43
    {0x5933f68ca078e30e, 0x2af81c591bac8f6a},  //  -42
    {0x6f80f42fc8971bd1, 0xb5b6236f6297b346},  //  -41
    {0x45b0989ddd5e7163, 0x1191d6259d9ed00c},  //  -40
    {0x571cbec554b60dbb, 0xd5f64baf0506840e},  //  -39
    {0x6ce3ee76a9e3912a, 0xcb73de9ac6482512},  //  -38
    {0x440e750a2a2e3aba, 0xbf286b20bbed172c},  //  -37
    {0x5512124cb4b9c969, 0x6ef285e8eae85cf6},  //  -36
    {0x6a5696dfe1e83bc3, 0xcaaf276325a27434},  //  -35
    {0x42761e4bed31255a, 0x5ead789df78588a0},  //  -34
    {0x5313a5dee87d6eb0, 0xf658d6c57566eac8},  //  -33
    {0x67d88f56a29cca5d, 0x33ef0c76d2c0a57a},  //  -32
    {0x40e7599625a1fe7a, 0x407567ca43b8676c},  //  -31
    {0x51212ffbaf0a7e18, 0xd092c1bcd4a68148},  //  -30
    {0x65697bfa9acd1d9f, 0x04b7722c09d0219a},  //  -29
    {0x7ec3daf941806506, 0xc5e54eb70c442a00},  //  -28
    {0x4f3a68dbc8f03f24, 0x3baf513267aa9a40},  //  -27
    {0x63090312bb2c4eed, 0x4a9b257f019540d0},  //  -26
    {0x7bcb43d769f762a8, 0x9d41eedec1fa9104},  //  -25
    {0x4d5f0a66a23a9da9, 0x6249354b393c9aa2},  //  -24
    {0x60b6cd004ac94513, 0xbadb829e078bc14a},  //  -23
    {0x78e480405d7b9658, 0xa9926345896eb19e},  //  -22
    {0x4b8ed0283a6d3df7, 0x69fb7e0b75e52f02},  //  -21
    {0x5e72843249088d75, 0x447a5d8e535e7ac4},  //  -20
    {0x760f253edb4ab0d2, 0x9598f4f1e8361974},  //  -19
    {0x49c97747490eae83, 0x9d7f99173121cfe8},  //  -18
    {0x5c3bd5191b525a24, 0x84df7f5cfd6a43e2},  //  -17
    {0x734aca5f6226f0ad, 0xa6175f343cc4d4da},  //  -16
    {0x480ebe7b9d58566c, 0x87ce9b80a5fb050a},  //  -15
    {0x5a126e1a84ae6c07, 0xa9c24260cf79c64c},  //  -14
    {0x709709a125da0709, 0x9432d2f9035837de},  //  -13
    {0x465e6604b7a84465, 0xfc9fc3dba21722ea},  //  -12
    {0x57f5ff85e592557f, 0x7bc7b4d28a9ceba6},  //  -11
    {0x6df37f675ef6eadf, 0x5ab9a2072d44268e},  //  -10
    {0x44b82fa09b5a52cb, 0x98b405447c4a981a},  //   -9
    {0x55e63b88c230e77e, 0x7ee106959b5d3e20},  //   -8
    {0x6b5fca6af2bd215e, 0x1e99483b02348da8},  //   -7
    {0x431bde82d7b634da, 0xd31fcd24e160d888},  //   -6
    {0x53e2d6238da3c211, 0x87e7c06e19b90eaa},  //   -5
    {0x68db8bac710cb295, 0xe9e1b089a0275256},  //   -4
    {0x4189374bc6a7ef9d, 0xb22d0e5604189376},  //   -3
    {0x51eb851eb851eb85, 0x1eb851eb851eb852},  //   -2
    {0x6666666666666666, 0x6666666666666668},  //   -1
    {0x4000000000000000, 0x0000000000000002},  //    0
    {0x5000000000000000, 0x0000000000000002},  //    1
    {0x6400000000000000, 0x0000000000000002},  //    2
    {0x7d00000000000000, 0x0000000000000002},  //    3
    {0x4e20000000000000, 0x0000000000000002},  //    4
    {0x61a8000000000000, 0x0000000000000002},  //    5
    {0x7a12000000000000, 0x0000000000000002},  //    6
    {0x4c4b400000000000, 0x0000000000000002},  //    7
    {0x5f5e100000000000, 0x0000000000000002},  //    8
    {0x7735940000000000, 0x0000000000000002},  //    9
    {0x4a817c8000000000, 0x0000000000000002},  //   10
    {0x5d21dba000000000, 0x0000000000000002},  //   11
    {0x746a528800000000, 0x0000000000000002},  //   12
    {0x48c2739500000000, 0x0000000000000002},  //   13
    {0x5af3107a40000000, 0x0000000000000002},  //   14
    {0x71afd498d0000000, 0x0000000000000002},  //   15
    {0x470de4df82000000, 0x0000000000000002},  //   16
    {0x58d15e1762800000, 0x0000000000000002},  //   17
    {0x6f05b59d3b200000, 0x0000000000000002},  //   18
    {0x4563918244f40000, 0x0000000000000002},  //   19
    {0x56bc75e2d6310000, 0x0000000000000002},  //   20
    {0x6c6b935b8bbd4000, 0x0000000000000002},  //   21
    {0x43c33c1937564800, 0x0000000000000002},  //   22
    {0x54b40b1f852bda00, 0x0000000000000002},  //   23
    {0x69e10de76676d080, 0x0000000000000002},  //   24
    {0x422ca8b0a00a4250, 0x0000000000000002},  //   25
    {0x52b7d2dcc80cd2e4, 0x0000000000000002},  //   26
    {0x6765c793fa10079d, 0x0000000000000002},  //   27
    {0x409f9cbc7c4a04c2, 0x2000000000000002},  //   28
    {0x50c783eb9b5c85f2, 0xa800000000000002},  //   29
    {0x64f964e68233a76f, 0x5200000000000002},  //   30
    {0x7e37be2022c0914b, 0x2680000000000002},  //   31
    {0x4ee2d6d415b85ace, 0xf810000000000002},  //   32
    {0x629b8c891b267182, 0xb614000000000002},  //   33
    {0x7b426fab61f00de3, 0x6399000000000002},  //   34
    {0x4d0985cb1d3608ae, 0x1e3fa00000000002},  //   35
    {0x604be73de4838ad9, 0xa5cf880000000002},  //   36
    {0x785ee10d5da46d90, 0x0f436a0000000002},  //   37
    {0x4b3b4ca85a86c47a, 0x098a224000000002},  //   38
    {0x5e0a1fd271287598, 0x8becaad000000002},  //   39
    {0x758ca7c70d7292fe, 0xaee7d58400000002},  //   40
    {0x4977e8dc68679bdf, 0x2d50e57280000002},  //   41
    {0x5bd5e313828182d6, 0xf8a51ecf20000002},  //   42
    {0x72cb5bd86321e38c, 0xb6ce6682e8000002},  //   43
    {0x47bf19673df52e37, 0xf2410011d1000002},  //   44
    {0x59aedfc10d7279c5, 0xeed1401645400002},  //   45
    {0x701a97b150cf1837, 0x6a85901bd6900002},  //   46
    {0x46109eced2816f22, 0xa2937a11661a0002},  //   47
    {0x5794c6828721caeb, 0x4b385895bfa08002},  //   48
    {0x6d79f82328ea3da6, 0x1e066ebb2f88a002},  //   49
    {0x446c3b15f9926687, 0xd2c40534fdb56402},  //   50
    {0x558749db77f70029, 0xc77506823d22bd02},  //   51
    {0x6ae91c5255f4c034, 0x39524822cc6b6c42},  //   52
    {0x42d1b1b375b8f820, 0xa3d36d15bfc323aa},  //   53
    {0x53861e2053273628, 0xccc8485b2fb3ec94},  //   54
    {0x6867a5a867f103b2, 0xfffa5a71fba0e7b8},  //   55
    {0x4140c78940f6a24f, 0xdffc78873d4490d4},  //   56
    {0x5190f96b91344ae3, 0xd7fb96a90c95b508},  //   57
    {0x65f537c675815d9c, 0xcdfa7c534fbb224a},  //   58
    {0x7f7285b812e1b504, 0x01791b6823a9eadc},  //   59
    {0x4fa793930bcd1122, 0x80ebb121164a32ca},  //   60
    {0x63917877cec0556b, 0x21269d695bdcbf7c},  //   61
    {0x7c75d695c2706ac5, 0xe97044c3b2d3ef5a},  //   62
    {0x4dc9a61d998642bb, 0xb1e62afa4fc47598},  //   63
    {0x613c0fa4ffe7d36a, 0x9e5fb5b8e3b592fe},  //   64
    {0x798b138e3fe1c845, 0x45f7a3271ca2f7be},  //   65
    {0x4bf6ec38e7ed1d2b, 0x4bbac5f871e5dad8},  //   66
    {0x5ef4a74721e86476, 0x1ea977768e5f518c},  //   67
    {0x76b1d118ea627d93, 0xa653d55431f725f0},  //   68
    {0x4a2f22af927d8e7c, 0x47f465549f3a77b6},  //   69
    {0x5cbaeb5b771cf21b, 0x59f17ea9c70915a4},  //   70
    {0x73e9a63254e42ea2, 0x306dde5438cb5b0c},  //   71
    {0x487207df750e9d25, 0x5e44aaf4a37f18e8},  //   72
    {0x5a8e89d75252446e, 0xb5d5d5b1cc5edf22},  //   73
    {0x71322c4d26e6d58a, 0x634b4b1e3f7696ea},  //   74
    {0x46bf5bb038504576, 0x7e0f0ef2e7aa1e52},  //   75
    {0x586f329c466456d4, 0x1d92d2afa194a5e6},  //   76
    {0x6e8aff4357fd6c89, 0x24f7875b89f9cf60},  //   77
    {0x4516df8a16fe63d5, 0xb71ab499363c219c},  //   78
    {0x565c976c9cbdfccb, 0x24e161bf83cb2a04},  //   79
    {0x6bf3bd47c3ed7bfd, 0xee19ba2f64bdf484},  //   80
    {0x4378564cda746d7e, 0xb4d0145d9ef6b8d2},  //   81
    {0x54566be0111188de, 0x6204197506b46708},  //   82
    {0x696c06d81555eb15, 0xfa851fd2486180ca},  //   83
    {0x41e384470d55b2ed, 0xbc9333e36d3cf07e},  //   84
    {0x525c6558d0ab1fa9, 0x2bb800dc488c2c9e},  //   85
    {0x66f37eaf04d5e793, 0x76a601135aaf37c4},  //   86
    {0x40582f2d6305b0bc, 0x2a27c0ac18ad82dc},  //   87
    {0x506e3af8bbc71ceb, 0x34b1b0d71ed8e392},  //   88
    {0x6489c9b6eab8e426, 0x01de1d0ce68f1c76},  //   89
    {0x7dac3c24a5671d2f, 0x8255a4502032e392},  //   90
    {0x4e8ba596e760723d, 0xb17586b2141fce3c},  //   91
    {0x622e8efca1388ecd, 0x1dd2e85e9927c1cc},  //   92
    {0x7aba32bbc986b280, 0x6547a2763f71b23e},  //   93
    {0x4cb45fb55df42f90, 0x3f4cc589e7a70f66},  //   94
    {0x5fe177a2b5713b74, 0x4f1ff6ec6190d340},  //   95
    {0x77d9d58b62cd8a51, 0x62e7f4a779f50810},  //   96
    {0x4ae825771dc07672, 0xddd0f8e8ac39250a},  //   97
    {0x5da22ed4e530940f, 0x95453722d7476e4c},  //   98
    {0x750aba8a1e7cb913, 0x7a9684eb8d1949e0},  //   99
    {0x4926b496530df3ac, 0x2c9e1313382fce2c},  //  100
    {0x5b7061bbe7d17097, 0x37c597d8063bc1b8},  //  101
    {0x724c7a2ae1c5ccbd, 0x05b6fdce07cab224},  //  102
    {0x476fcc5acd1b9ff6, 0x23925ea0c4deaf58},  //  103
    {0x594bbf71806287f3, 0xac76f648f6165b2c},  //  104
    {0x6f9eaf4de07b29f0, 0x9794b3db339bf1f8},  //  105
    {0x45c32d90ac4cfa36, 0x5ebcf0690041773c},  //  106
    {0x5733f8f4d76038c3, 0xf66c2c834051d50a},  //  107
    {0x6d00f7320d3846f4, 0xf40737a410664a4c},  //  108
    {0x44209a7f48432c59, 0x188482c68a3fee70},  //  109
    {0x5528c11f1a53f76f, 0x5ea5a3782ccfea0c},  //  110
    {0x6a72f166e0e8f54b, 0x364f0c563803e48e},  //  111
    {0x4287d6e04c91994f, 0x01f167b5e3026eda},  //  112
    {0x5329cc985fb5ffa2, 0xc26dc1a35bc30a90},  //  113
    {0x67f43fbe77a37f8b, 0x7309320c32b3cd32},  //  114
    {0x40f8a7d70ac62fb7, 0x27e5bf479fb06040},  //  115
    {0x5136d1cccd77bba4, 0xf1df2f19879c7850},  //  116
    {0x6584864000d5aa8e, 0x2e56fadfe9839664},  //  117
    {0x7ee5a7d0010b1531, 0xb9ecb997e3e47bfc},  //  118
    {0x4f4f88e200a6ed3f, 0x1433f3feee6ecd7e},  //  119
    {0x63236b1a80d0a88e, 0xd940f0feaa0a80de},  //  120
    {0x7bec45e12104d2b2, 0x8f912d3e548d2114},  //  121
    {0x4d73abacb4a303af, 0x99babc46f4d834ae},  //  122
    {0x60d09697e1cbc49b, 0x80296b58b20e41d8},  //  123
    {0x7904bc3dda3eb5c2, 0x6033c62ede91d24e},  //  124
    {0x4ba2f5a6a8673199, 0x7c205bdd4b1b2372},  //  125
    {0x5e8bb3105280fdff, 0xdb2872d49de1ec4e},  //  126
    {0x762e9fd467213d7f, 0xd1f28f89c55a6760},  //  127
    {0x49dd23e4c074c66f, 0xe33799b61b58809c},  //  128
    {0x5c546cddf091f80b, 0xdc058023a22ea0c4},  //  129
    {0x736988156cb6760e, 0xd306e02c8aba48f4},  //  130
    {0x4821f50d63f209c9, 0x43e44c1bd6b46d98},  //  131
    {0x5a2a7250bcee8c3b, 0x94dd5f22cc6188fe},  //  132
    {0x70b50ee4ec2a2f4a, 0x7a14b6eb7f79eb3e},  //  133
    {0x4671294f139a5d8e, 0x8c4cf2532fac3308},  //  134
    {0x580d73a2d880f4f2, 0x2f602ee7fb973fc8},  //  135
    {0x6e10d08b8ea1322e, 0xbb383aa1fa7d0fba},  //  136
    {0x44ca82573924bf5d, 0x350324a53c8e29d6},  //  137
    {0x55fd22ed076def34, 0x8243edce8bb1b44a},  //  138
    {0x6b7c6ba849496b01, 0xa2d4e9422e9e215c},  //  139
    {0x432dc3492dcde2e1, 0x05c511c95d22d4da},  //  140
    {0x53f9341b79415b99, 0x4736563bb46b8a10},  //  141
    {0x68f781225791b27f, 0x9903ebcaa1866c94},  //  142
    {0x419ab0b576bb0f8f, 0xbfa2735ea4f403de},  //  143
    {0x52015ce2d469d373, 0xaf8b10364e3104d4},  //  144
    {0x6681b41b89844850, 0x9b6dd443e1bd4608},  //  145
    {0x4011109135f2ad32, 0x6124a4aa6d164bc6},  //  146
    {0x501554b5836f587e, 0xf96dcdd5085bdeb8},  //  147
    {0x641aa9e2e44b2e9e, 0xb7c9414a4a72d664},  //  148
    {0x7d21545b9d5dfa46, 0x65bb919cdd0f8bfe},  //  149
    {0x4e34d4b9425abc6b, 0xff953b020a29b77e},  //  150
    {0x61c209e792f16b86, 0xff7a89c28cb4255e},  //  151
    {0x7a328c6177adc668, 0xbf592c332fe12eb6},  //  152
    {0x4c5f97bceacc9c01, 0x7797bb9ffdecbd32},  //  153
    {0x5f777dac257fc301, 0xd57daa87fd67ec7e},  //  154
    {0x77555d172edfb3c2, 0x4add1529fcc1e79e},  //  155
    {0x4a955a2e7d4bd059, 0x6eca2d3a3df930c2},  //  156
    {0x5d3ab0ba1c9ec46f, 0xca7cb888cd777cf4},  //  157
    {0x74895ce8a3c6758b, 0xbd1be6ab00d55c30},  //  158
    {0x48d5da11665c0977, 0x5631702ae085599e},  //  159
    {0x5b0b5095bff30bd5, 0x2bbdcc3598a6b006},  //  160
    {0x71ce24bb2fefceca, 0x76ad3f42fed05c06},  //  161
    {0x4720d6f4fdf5e13e, 0x8a2c4789df423984},  //  162
    {0x58e90cb23d73598e, 0x2cb7596c5712c7e6},  //  163
    {0x6f234fdeccd02ff1, 0xb7e52fc76cd779de},  //  164
    {0x457611eb40021df7, 0x12ef3ddca406ac2c},  //  165
    {0x56d396661002a574, 0xd7ab0d53cd085736},  //  166
    {0x6c887bff94034ed2, 0x0d95d0a8c04a6d04},  //  167
    {0x43d54d7fbc821143, 0x487da269782e8422},  //  168
    {0x54caa0dfaba29594, 0x1a9d0b03d63a252a},  //  169
    {0x69fd4917968b3af9, 0x21444dc4cbc8ae76},  //  170
    {0x423e4daebe1704db, 0xb4cab09aff5d6d0a},  //  171
    {0x52cde11a6d9cc612, 0xa1fd5cc1bf34c84c},  //  172
    {0x678159610903f797, 0x4a7cb3f22f01fa5e},  //  173
    {0x40b0d7dca5a27abe, 0x8e8df0775d613c7c},  //  174
    {0x50dd0dd3cf0b196e, 0x32316c9534b98b9a},  //  175
    {0x65145148c2cddfc9, 0xbebdc7ba81e7ee80},  //  176
    {0x7e59659af38157bc, 0x2e6d39a92261ea20},  //  177
    {0x4ef7df80d830d6d5, 0x9d044409b57d3254},  //  178
    {0x62b5d7610e3d0c8b, 0x0445550c22dc7eea},  //  179
    {0x7b634d3951cc4fad, 0xc556aa4f2b939ea4},  //  180
    {0x4d1e1043d31fb1cc, 0x9b562a717b3c4326},  //  181
    {0x60659454c7e79e3f, 0xc22bb50dda0b53f0},  //  182
    {0x787ef969f9e185cf, 0xb2b6a251508e28ec},  //  183
    {0x4b4f5be23c2cf3a1, 0xcfb22572d258d994},  //  184
    {0x5e2332dacb38308a, 0x439eaecf86ef0ff8},  //  185
    {0x75abff917e063cac, 0xd4865a8368aad3f6},  //  186
    {0x498b7fbaeec3e5ec, 0x04d3f892216ac47a},  //  187
    {0x5bee5fa9aa74df67, 0x0608f6b6a9c57598},  //  188
    {0x72e9f79415121740, 0xc78b34645436d2fe},  //  189
    {0x47d23abc8d2b4e88, 0x7cb700beb4a243e0},  //  190
    {0x59c6c96bb076222a, 0x9be4c0ee61cad4d8},  //  191
    {0x70387bc69c93aab5, 0x42ddf129fa3d8a0c},  //  192
    {0x46234d5c21dc4ab1, 0x49cab6ba3c667648},  //  193
    {0x57ac20b32a535d5d, 0x9c3d6468cb8013da},  //  194
    {0x6d9728dff4e834b5, 0x034cbd82fe6018d0},  //  195
    {0x447e798bf91120f1, 0x220ff671defc0f82},  //  196
    {0x559e17eef755692d, 0x6a93f40e56bb1362},  //  197
    {0x6b059deab52ac378, 0xc538f111ec69d83c},  //  198
    {0x42e382b2b13aba2b, 0x7b4396ab33c22726},  //  199
    {0x539c635f5d8968b6, 0x5a147c5600b2b0ee},  //  200
    {0x68837c3734ebc2e3, 0xf0999b6b80df5d2a},  //  201
    {0x41522da2811359ce, 0x76600123308b9a3a},  //  202
    {0x51a6b90b21583042, 0x13f8016bfcae80ca},  //  203
    {0x6610674de9ae3c52, 0x98f601c6fbda20fc},  //  204
    {0x7f9481216419cb67, 0x3f338238bad0a93a},  //  205
    {0x4fbcd0b4de901f20, 0x8780316374c269c4},  //  206
    {0x63ac04e2163426e8, 0xa9603dbc51f30436},  //  207
    {0x7c97061a9bc130a2, 0xd3b84d2b666fc542},  //  208
    {0x4dde63d0a158be65, 0xc453303b2005db4a},  //  209
    {0x6155fcc4c9aeedff, 0x3567fc49e807521c},  //  210
    {0x79ab7bf5fc1aa97f, 0x02c1fb5c620926a2},  //  211
    {0x4c0b2d79bd90a9ef, 0x61b93d19bd45b826},  //  212
    {0x5f0df8d82cf4d46b, 0x3a278c602c972630},  //  213
    {0x76d1770e38320986, 0x08b16f7837bcefba},  //  214
    {0x4a42ea68e31f45f3, 0xc56ee5ab22d615d6},  //  215
    {0x5cd3a5031be71770, 0xb6ca9f15eb8b9b4a},  //  216
    {0x74088e43e2e0dd4c, 0xe47d46db666e821c},  //  217
    {0x488558ea6dcc8a50, 0x0ece4c4920051152},  //  218
    {0x5aa6af25093face4, 0x1281df5b680655a6},  //  219
    {0x71505aee4b8f981d, 0x172257324207eb10},  //  220
    {0x46d238d4ef39bf12, 0x2e75767f6944f2ea},  //  221
    {0x5886c70a2b082ed6, 0xba12d41f43962fa4},  //  222
    {0x6ea878ccb5ca3a8c, 0x68978927147bbb8e},  //  223
    {0x45294b7ff19e6497, 0xc15eb5b86ccd5538},  //  224
    {0x56739e5fee05fdbd, 0xb1b663268800aa86},  //  225
    {0x6c1085f7e9877d2d, 0x1e23fbf02a00d528},  //  226
    {0x438a53baf1f4ae3c, 0x32d67d761a40853a},  //  227
    {0x546ce8a9ae71d9cb, 0x3f8c1cd3a0d0a688},  //  228
    {0x698822d41a0e503e, 0x0f6f24088904d02a},  //  229
    {0x41f515c49048f226, 0xc9a5768555a3021a},  //  230
    {0x52725b35b45b2eb0, 0x7c0ed426ab0bc2a0},  //  231
    {0x670ef2032171fa5c, 0x9b12893055ceb348},  //  232
    {0x40695741f4e73c79, 0xe0eb95be35a1300e},  //  233
    {0x5083ad1272210b98, 0x59267b2dc3097c10},  //  234
    {0x64a498570ea94e7e, 0x6f7019f933cbdb14},  //  235
    {0x7dcdbe6cd253a21e, 0x0b4c207780bed1da},  //  236
    {0x4ea0970403744552, 0xc70f944ab0774328},  //  237
    {0x6248bcc5045156a7, 0x78d3795d5c9513f2},  //  238
    {0x7adaebf64565ac51, 0x570857b4b3ba58ee},  //  239
    {0x4cc8d379eb5f8bb2, 0xd66536d0f0547796},  //  240
    {0x5ffb085866376e9f, 0x8bfe84852c69957a},  //  241
    {0x77f9ca6e7fc54a47, 0x6efe25a67783fada},  //  242
    {0x4afc1e850fdb4e6c, 0xa55ed7880ab27cc8},  //  243
    {0x5dbb262653d22207, 0xceb68d6a0d5f1bfa},  //  244
    {0x7529efafe8c6aa89, 0xc26430c490b6e2f8},  //  245
    {0x493a35cdf17c2a96, 0x197e9e7ada724ddc},  //  246
    {0x5b88c3416ddb353b, 0x9fde4619910ee152},  //  247
    {0x726af411c952028a, 0x87d5d79ff55299a6},  //  248
    {0x4782d88b1dd34196, 0x94e5a6c3f953a008},  //  249
    {0x59638eade54811fc, 0x3a1f1074f7a8880a},  //  250
    {0x6fbc72595e9a167b, 0x48a6d4923592aa0c},  //  251
    {0x45d5c777db204e0d, 0x0d6844db617baa48},  //  252
    {0x574b3955d1e86190, 0x50c2561239da94da},  //  253
    {0x6d1e07ab466279f4, 0x64f2eb96c8513a10},  //  254
    {0x4432c4cb0bfd8c38, 0xbf17d33e3d32c44a},  //  255
    {0x553f75fdcefcef46, 0xeeddc80dcc7f755c},  //  256
    {0x6a8f537d42bc2b18, 0xaa953a113f9f52b4},  //  257
    {0x4299942e49b59aef, 0x6a9d444ac7c393b0},  //  258
    {0x533ff939dc2301ab, 0x4544955d79b4789c},  //  259
    {0x680ff788532bc216, 0x1695bab4d82196c4},  //  260
    {0x4109fab533fb594d, 0xce1d94b10714fe3a},  //  261
    {0x514c796280fa2fa1, 0x41a4f9dd48da3dc8},  //  262
    {0x659f97bb2138bb89, 0x920e38549b10cd3a},  //  263
    {0x7f077da9e986ea6b, 0xf691c669c1d5008a},  //  264
    {0x4f64ae8a31f45283, 0x7a1b1c0219252056},  //  265
    {0x633dda2cbe716724, 0x58a1e3029f6e686c},  //  266
    {0x7c0d50b7ee0dc0ed, 0x6eca5bc3474a0286},  //  267
    {0x4d885272f4c89894, 0x653e795a0c8e4194},  //  268
    {0x60ea670fb1fabeb9, 0x7e8e17b08fb1d1fa},  //  269
    {0x792500d39e796e67, 0xde319d9cb39e4678},  //  270
    {0x4bb72084430be500, 0xeadf0281f042ec0a},  //  271
    {0x5ea4e8a553cede41, 0x2596c3226c53a70e},  //  272
    {0x764e22cea8c295d1, 0x6efc73eb076890d0},  //  273
    {0x49f0d5c129799da2, 0xe55dc872e4a15a82},  //  274
    {0x5c6d0b3173d8050b, 0x9eb53a8f9dc9b122},  //  275
    {0x73884dfdd0ce064e, 0x86628933853c1d6c},  //  276
    {0x483530bea280c3f1, 0x13fd95c033459264},  //  277
    {0x5a427cee4b20f4ed, 0x58fcfb304016f6fc},  //  278
    {0x70d31c29dde93228, 0xaf3c39fc501cb4ba},  //  279
    {0x4683f19a2ab1bf59, 0x6d85a43db211f0f6},  //  280
    {0x5824ee00b55e2f2f, 0xc8e70d4d1e966d32},  //  281
    {0x6e2e2980e2b5bafb, 0xbb20d0a0663c087e},  //  282
    {0x44dcd9f08db194dd, 0x54f482643fe58550},  //  283
    {0x5614106cb11dfa14, 0xaa31a2fd4fdee6a4},  //  284
    {0x6b991487dd657899, 0xd4be0bbca3d6a04c},  //  285
    {0x433facd4ea5f6b60, 0x24f6c755e6662430},  //  286
    {0x540f980a24f74638, 0x2e34792b5fffad3c},  //  287
    {0x69137e0cae3517c6, 0x39c1977637ff988a},  //  288
    {0x41ac2ec7ece12edb, 0xe418fea9e2ffbf56},  //  289
    {0x52173a79e8197a92, 0xdd1f3e545bbfaf2c},  //  290
    {0x669d0918621fd937, 0x94670de972af9af6},  //  291
    {0x402225af3d53e7c2, 0xbcc068b1e7adc0da},  //  292
    {0x502aaf1b0ca8e1b3, 0x6bf082de61993110},  //  293
    {0x64355ae1cfd31a20, 0x46eca395f9ff7d54},  //  294
    {0x7d42b19a43c7e0a8, 0x58a7cc7b787f5caa},  //  295
    {0x4e49af006a5cec69, 0x3768dfcd2b4f99ea},  //  296
    {0x61dc1ac084f42783, 0x854317c076238066},  //  297
    {0x7a532170a6313164, 0x6693ddb093ac607e},  //  298
    {0x4c73f4e667debede, 0xc01c6a8e5c4bbc50},  //  299
    {0x5f90f22001d66e96, 0x70238531f35eab62},  //  300
    {0x77752ea8024c0a3c, 0x0c2c667e7036563c},  //  301
    {0x4aa93d29016f8665, 0x879bc00f0621f5e6},  //  302
    {0x5d538c7341cb67fe, 0xe982b012c7aa735e},  //  303
    {0x74a86f90123e41fe, 0xa3e35c1779951036},  //  304
    {0x48e945ba0b66e93f, 0x266e198eabfd2a22},  //  305
    {0x5b2397288e40a38e, 0xf0099ff256fc74aa},  //  306
    {0x71ec7cf2b1d0cc72, 0xac0c07eeecbb91d4},  //  307
    {0x4733ce17af227fc7, 0xab8784f553f53b26},  //  308
    {0x5900c19d9aeb1fb9, 0x96696632a8f289ee},  //  309
    {0x6f40f20501a5e7a7, 0xfc03bfbf532f2c6a},  //  310
    {0x458897432107b0c8, 0xfd8257d793fd7bc2},  //  311
    {0x56eabd13e9499cfb, 0x3ce2edcd78fcdab2},  //  312
    {0x6ca56c58e39c043a, 0x0c1ba940d73c1160},  //  313
    {0x43e763b78e4182a4, 0x479149c886858adc},  //  314
    {0x54e13ca571d1e34d, 0x59759c3aa826ed92},  //  315
    {0x6a198bcece465c20, 0xafd303495230a8f6},  //  316
    {0x424ff76140ebf994, 0x6de3e20dd35e699a},  //  317
    {0x52e3f5399126f7f9, 0x895cda9148360402},  //  318
    {0x679cf287f570b5f7, 0xebb411359a438502},  //  319
    {0x40c21794f96671ba, 0xf3508ac1806a3322},  //  320
    {0x50f29d7a37c00e29, 0xb024ad71e084bfea},  //  321
    {0x652f44d8c5b011b4, 0x1c2dd8ce58a5efe4},  //  322
    {0x7e7b160ef71c1621, 0x23394f01eecf6bdc},  //  323
    {0x4f0cedc95a718dd4, 0xb603d1613541a36a},  //  324
};

// Computes 128-bit result of multiplication of two 64-bit unsigned integers.
auto umul128(uint64_t x, uint64_t y) noexcept -> uint128_t {
#ifdef __SIZEOF_INT128__
  return uint128_t(x) * y;
#else
  constexpr uint64_t mask = ~uint32_t();

  uint64_t a = x >> 32;
  uint64_t b = x & mask;
  uint64_t c = y >> 32;
  uint64_t d = y & mask;

  uint64_t ac = a * c;
  uint64_t bc = b * c;
  uint64_t ad = a * d;
  uint64_t bd = b * d;

  uint64_t intermediate = (bd >> 32) + (ad & mask) + (bc & mask);

  return {ac + (intermediate >> 32) + (ad >> 32) + (bc >> 32),
          (intermediate << 32) + (bd & mask)};
#endif  // __SIZEOF_INT128__
}

// Computes upper 64 bits of multiplication of pow10 and scaled_sig with
// modified round-to-odd rounding of the result,
// where pow10 = (pow10_hi << 64) | pow10_lo.
auto umul192_upper64_modified(uint64_t pow10_hi, uint64_t pow10_lo,
                              uint64_t scaled_sig) noexcept -> uint64_t {
  uint64_t x_hi = uint64_t(umul128(pow10_lo, scaled_sig) >> 64);
  uint128_t y = umul128(pow10_hi, scaled_sig) + x_hi;
  uint64_t z = uint64_t(y >> 1);
  constexpr uint64_t mask = (uint64_t(1) << 63) - 1;
  // OR with 1 if z is not divisible by 2**63.
  return uint64_t(y >> 64) | (((z & mask) + mask) >> 63);
}

// Converts value in the range [0, 100) to a string. GCC generates a bit better
// code when value is pointer-size (https://www.godbolt.org/z/5fEPMT1cc).
inline auto digits2(size_t value) noexcept -> const char* {
  // Align data since unaligned access may be slower when crossing a
  // hardware-specific boundary.
  alignas(2) static const char data[] =
      "0001020304050607080910111213141516171819"
      "2021222324252627282930313233343536373839"
      "4041424344454647484950515253545556575859"
      "6061626364656667686970717273747576777879"
      "8081828384858687888990919293949596979899";
  return &data[value * 2];
}

// The idea of branchless trailing zero removal is by Alexander Bolz.
const char num_trailing_zeros[] =
    "\2\0\0\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\0\0"
    "\1\0\0\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\0\0"
    "\1\0\0\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\0\0"
    "\1\0\0\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\0\0"
    "\1\0\0\0\0\0\0\0\0\0\1\0\0\0\0\0\0\0\0\0";

struct div_mod_result {
  uint32_t div;
  uint32_t mod;
};

// Returns {value / 100, value % 100} correct for values of up to num_digits
// decimal digits where num_digits should be 3 or 4.
template <int num_digits>
inline auto divmod100(uint32_t value) noexcept -> div_mod_result {
  static_assert(num_digits == 3 || num_digits == 4, "wrong number of digits");
  constexpr int exp = 19;  // 19 is faster than 12 for 3 digits.
  assert(value < (num_digits == 3 ? 1'000 : 10'000));
  constexpr int sig = (1 << exp) / 100 + 1;
  uint32_t div = (value * sig) >> exp;  // value / 100
  return {div, value - div * 100};
}

inline void write2digits(char* buffer, uint32_t value) noexcept {
  memcpy(buffer, digits2(value), 2);
}

// Writes 4 digits and removes trailing zeros.
auto write4digits(char* buffer, uint32_t value) noexcept -> char* {
  auto [aa, bb] = divmod100<4>(value);
  write2digits(buffer + 0, aa);
  write2digits(buffer + 2, bb);
  return buffer + 4 - num_trailing_zeros[bb] -
         (bb == 0) * num_trailing_zeros[aa];
}

// Writes a significand consisting of 16 or 17 decimal digits and removes
// trailing zeros.
auto write_significand(char* buffer, uint64_t value) noexcept -> char* {
  // Each digits is denoted by a letter so value is abbccddeeffgghhii where
  // digit a can be zero.
  uint32_t abbccddee = uint32_t(value / 100'000'000);
  uint32_t ffgghhii = uint32_t(value % 100'000'000);
  uint32_t abbcc = abbccddee / 10'000;
  uint32_t ddee = abbccddee % 10'000;
  uint32_t abb = abbcc / 100;
  uint32_t cc = abbcc % 100;
  auto [a, bb] = divmod100<3>(abb);

  *buffer = char('0' + a);
  buffer += a != 0;
  write2digits(buffer + 0, bb);
  write2digits(buffer + 2, cc);
  buffer += 4;

  if (ffgghhii == 0) {
    if (ddee != 0) return write4digits(buffer, ddee);
    return buffer - num_trailing_zeros[cc] - (cc == 0) * num_trailing_zeros[bb];
  }
  auto [dd, ee] = divmod100<4>(ddee);
  uint32_t ffgg = ffgghhii / 10'000;
  uint32_t hhii = ffgghhii % 10'000;
  auto [ff, gg] = divmod100<4>(ffgg);
  write2digits(buffer + 0, dd);
  write2digits(buffer + 2, ee);
  write2digits(buffer + 4, ff);
  write2digits(buffer + 6, gg);
  if (hhii != 0) return write4digits(buffer + 8, hhii);
  return buffer + 8 - num_trailing_zeros[gg] -
         (gg == 0) * num_trailing_zeros[ff];
}

// Writes the decimal FP number dec_sig * 10**dec_exp to buffer.
void write(char* buffer, uint64_t dec_sig, int dec_exp) noexcept {
  dec_exp += 15 + (dec_sig >= uint64_t(1e16));

  char* start = buffer;
  buffer = write_significand(buffer + 1, dec_sig);
  start[0] = start[1];
  start[1] = '.';

  *buffer++ = 'e';
  char sign = '+';
  if (dec_exp < 0) {
    sign = '-';
    dec_exp = -dec_exp;
  }
  *buffer++ = sign;
  auto [a, bb] = divmod100<3>(uint32_t(dec_exp));
  *buffer = char('0' + a);
  buffer += dec_exp >= 100;
  write2digits(buffer, bb);
  buffer[2] = '\0';
}

}  // namespace

namespace zmij {

void dtoa(double value, char* buffer) noexcept {
  uint64_t bits = 0;
  memcpy(&bits, &value, sizeof(value));
  *buffer = '-';
  buffer += bits >> 63;

  constexpr int num_sig_bits = std::numeric_limits<double>::digits - 1;
  constexpr int exp_mask = 0x7ff;
  int bin_exp = int(bits >> num_sig_bits) & exp_mask;

  constexpr uint64_t implicit_bit = uint64_t(1) << num_sig_bits;
  uint64_t bin_sig = bits & (implicit_bit - 1);  // binary significand

  bool regular = bin_sig != 0;
  if (((bin_exp + 1) & exp_mask) <= 1) [[unlikely]] {
    if (bin_exp != 0) {
      memcpy(buffer, bin_sig == 0 ? "inf" : "nan", 4);
      return;
    }
    if (bin_sig == 0) {
      memcpy(buffer, "0", 2);
      return;
    }
    // Handle subnormals.
    bin_sig ^= implicit_bit;
    bin_exp = 1;
    regular = true;
  }
  bin_sig ^= implicit_bit;
  bin_exp -= num_sig_bits + 1023;  // Remove the exponent bias.

  // Handle small integers.
  if ((bin_exp < 0) & (bin_exp >= -num_sig_bits)) {
    uint64_t f = bin_sig >> -bin_exp;
    if (f << -bin_exp == bin_sig) return write(buffer, f, 0);
  }

  // Shift the significand so that boundaries are integer.
  uint64_t bin_sig_shifted = bin_sig << 2;

  // Compute the shifted boundaries of the rounding interval (Rv).
  uint64_t lower = bin_sig_shifted - (regular + 1);
  uint64_t upper = bin_sig_shifted + 2;

  // log10_3_over_4_sig = round(log10(3/4) * 2**log10_2_exp)
  constexpr int log10_3_over_4_sig = -131'008;
  // log10_2_sig = round(log10(2) * 2**log10_2_exp)
  constexpr int log10_2_sig = 315'653;
  constexpr int log10_2_exp = 20;

  // Compute the decimal exponent as floor(log10(2**bin_exp)) if regular or
  // floor(log10(3/4 * 2**bin_exp)) otherwise, without branching.
  assert(bin_exp >= -1334 && bin_exp <= 2620);
  int dec_exp =
      (bin_exp * log10_2_sig + !regular * log10_3_over_4_sig) >> log10_2_exp;

  constexpr int dec_exp_min = -292;
  auto [pow10_hi, pow10_lo] = pow10_significands[-dec_exp - dec_exp_min];

  // log2_pow10_sig = round(log2(10) * 2**log2_pow10_exp) + 1
  constexpr int log2_pow10_sig = 217'707, log2_pow10_exp = 16;

  assert(dec_exp >= -350 && dec_exp <= 350);
  // pow10_bin_exp = floor(log2(10**-dec_exp))
  int pow10_bin_exp = -dec_exp * log2_pow10_sig >> log2_pow10_exp;
  // pow10 = ((pow10_hi << 63) | pow10_lo) * 2**(pow10_bin_exp - 126 + 1)

  // Shift to ensure the intermediate result in umul192_upper64_modified has
  // a fixed 128-bit fractional width. For example, 3 * 2**59 and 3 * 2**60
  // both have dec_exp = 2 and dividing them by 10**dec_exp would have the
  // decimal point in different (bit) positions without the shift:
  //   3 * 2**59 / 100 = 1.72...e+16 (shift = 3)
  //   3 * 2**60 / 100 = 3.45...e+16 (shift = 4)
  int shift = bin_exp + pow10_bin_exp + 2;

  // Compute the estimates of lower and upper bounds of the rounding interval
  // by multiplying them by the power of 10 and applying modified rounding.
  uint64_t bin_sig_lsb = bin_sig & 1;
  lower = umul192_upper64_modified(pow10_hi, pow10_lo, lower << shift) +
          bin_sig_lsb;
  upper = umul192_upper64_modified(pow10_hi, pow10_lo, upper << shift) -
          bin_sig_lsb;

  // The idea of using a single shorter candidate is by Cassio Neri.
  // It is less or equal to the upper bound by construction.
  uint64_t shorter = 10 * ((upper >> 2) / 10);
  if ((shorter << 2) >= lower) return write(buffer, shorter, dec_exp);

  uint64_t scaled_sig =
      umul192_upper64_modified(pow10_hi, pow10_lo, bin_sig_shifted << shift);
  uint64_t dec_sig_under = scaled_sig >> 2;
  uint64_t dec_sig_over = dec_sig_under + 1;

  // Pick the closest of dec_sig_under and dec_sig_over and check if it's in
  // the rounding interval.
  int64_t cmp = int64_t(scaled_sig - ((dec_sig_under + dec_sig_over) << 1));
  bool under_closer = cmp < 0 || (cmp == 0 && (dec_sig_under & 1) == 0);
  bool under_in = (dec_sig_under << 2) >= lower;
  write(buffer, (under_closer & under_in) ? dec_sig_under : dec_sig_over,
        dec_exp);
}

}  // namespace zmij
