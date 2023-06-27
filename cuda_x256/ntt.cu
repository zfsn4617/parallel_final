#include "functions.h"

__constant__ int bit_reverse[256];
__constant__ int omega_pow_use_rank[512];

void Initialize_NTT()
{
    cudaError_t err;
    int bit_reverse_cpu[256] = {
        0, 128, 64, 192, 32, 160, 96, 224, 16, 144, 80, 208, 48, 176, 112, 240,
        8, 136, 72, 200, 40, 168, 104, 232, 24, 152, 88, 216, 56, 184, 120, 248,
        4, 132, 68, 196, 36, 164, 100, 228, 20, 148, 84, 212, 52, 180, 116, 244,
        12, 140, 76, 204, 44, 172, 108, 236, 28, 156, 92, 220, 60, 188, 124, 252,
        2, 130, 66, 194, 34, 162, 98, 226, 18, 146, 82, 210, 50, 178, 114, 242,
        10, 138, 74, 202, 42, 170, 106, 234, 26, 154, 90, 218, 58, 186, 122, 250,
        6, 134, 70, 198, 38, 166, 102, 230, 22, 150, 86, 214, 54, 182, 118, 246,
        14, 142, 78, 206, 46, 174, 110, 238, 30, 158, 94, 222, 62, 190, 126, 254,
        1, 129, 65, 193, 33, 161, 97, 225, 17, 145, 81, 209, 49, 177, 113, 241,
        9, 137, 73, 201, 41, 169, 105, 233, 25, 153, 89, 217, 57, 185, 121, 249,
        5, 133, 69, 197, 37, 165, 101, 229, 21, 149, 85, 213, 53, 181, 117, 245,
        13, 141, 77, 205, 45, 173, 109, 237, 29, 157, 93, 221, 61, 189, 125, 253,
        3, 131, 67, 195, 35, 163, 99, 227, 19, 147, 83, 211, 51, 179, 115, 243,
        11, 139, 75, 203, 43, 171, 107, 235, 27, 155, 91, 219, 59, 187, 123, 251,
        7, 135, 71, 199, 39, 167, 103, 231, 23, 151, 87, 215, 55, 183, 119, 247,
        15, 143, 79, 207, 47, 175, 111, 239, 31, 159, 95, 223, 63, 191, 127, 255};
    cudaMemcpyToSymbol(bit_reverse, bit_reverse_cpu, sizeof(bit_reverse_cpu));
    err = cudaGetLastError();
    printf("%s\n", cudaGetErrorString(err));
    int omega_pow_use_rank_cpu[512] = {
        0, 8380416, 4808194, 3572223, 4614810, 4618904, 3765607, 3761513,
        2883726, 5178987, 5178923, 3145678, 5496691, 3201430, 3201494, 5234739,
        6250525, 7822959, 601683, 7375178, 2682288, 1221177, 4837932, 4615550,
        2129892, 557458, 7778734, 1005239, 5698129, 7159240, 3542485, 3764867,
        7044481, 4795319, 4317364, 2453983, 4855975, 6096684, 1674615, 6666122,
        7703827, 642628, 3370349, 1460718, 7946292, 2815639, 2663378, 5152541,
        1335936, 3585098, 4063053, 5926434, 3524442, 2283733, 6705802, 1714295,
        676590, 7737789, 5010068, 6919699, 434125, 5564778, 5717039, 3227876,
        3241972, 7823561, 2740543, 4623627, 394148, 1858416, 7220542, 4805951,
        4018989, 3192354, 5197539, 6663429, 7284949, 2917338, 3110818, 3415069,
        2156050, 4510100, 4793971, 1935799, 928749, 5034454, 3704823, 817536,
        2071829, 2897314, 3602218, 4430364, 3506380, 1853806, 6279007, 1759347,
        5138445, 556856, 5639874, 3756790, 7986269, 6522001, 1159875, 3574466,
        4361428, 5188063, 3182878, 1716988, 1095468, 5463079, 5269599, 4965348,
        6224367, 3870317, 3586446, 6444618, 7451668, 3345963, 4675594, 7562881,
        6308588, 5483103, 4778199, 3950053, 4874037, 6526611, 2101410, 6621070,
        6644104, 6067579, 4183372, 2461387, 6852351, 2236726, 4222329, 7080401,
        5183169, 5697147, 4528402, 3901472, 169688, 8031605, 6352299, 5801164,
        5130263, 7921254, 3121440, 7759253, 1148858, 6458164, 5569126, 4182915,
        4213992, 5604662, 5307408, 5454601, 3334383, 1011223, 4564692, 2391089,
        8145010, 4912752, 5157610, 1317678, 7897768, 6635910, 7270901, 6018354,
        6392603, 2778788, 5744944, 7153756, 565603, 327848, 2508980, 1787943,
        3258457, 653275, 274060, 3035980, 5418153, 3818627, 2983781, 3482206,
        4892034, 7023969, 7102792, 5006167, 2462444, 6026202, 6442847, 2254727,
        1736313, 2312838, 4197045, 5919030, 1528066, 6143691, 4158088, 1300016,
        3197248, 2683270, 3852015, 4478945, 8210729, 348812, 2028118, 2579253,
        3250154, 459163, 5258977, 621164, 7231559, 1922253, 2811291, 4197502,
        4166425, 2775755, 3073009, 2925816, 5046034, 7369194, 3815725, 5989328,
        235407, 3467665, 3222807, 7062739, 482649, 1744507, 1109516, 2362063,
        1987814, 5601629, 2635473, 1226661, 7814814, 8052569, 5871437, 6592474,
        5121960, 7727142, 8106357, 5344437, 2962264, 4561790, 5396636, 4898211,
        3488383, 1356448, 1277625, 3374250, 5917973, 2354215, 1937570, 6125690,
        1921994, 7826699, 1182243, 5732423, 6607829, 781875, 5925040, 507927,
        1310261, 214880, 5607817, 4399818, 1239911, 5256655, 5926272, 6757063,
        6341273, 140244, 2296397, 4357667, 2387513, 3974485, 4969849, 1393159,
        5382198, 7009900, 1935420, 2028038, 12417, 3014420, 4423473, 1179613,
        4908348, 3105558, 7743490, 8041997, 1727088, 7648983, 4829411, 724804,
        613238, 770441, 5720009, 6764887, 6084318, 6187330, 8352605, 2374402,
        7561656, 4949981, 4663471, 5767564, 268456, 3531229, 3768948, 1476985,
        8291116, 11879, 6924527, 3369273, 5184741, 2926054, 6783595, 5637006,
        7921677, 7872272, 87208, 5370669, 4146264, 1900052, 250446, 7192532,
        2218467, 5016875, 8321269, 5811406, 4541938, 6195333, 7371052, 2105286,
        1879878, 6866265, 4423672, 7630840, 4768667, 3773731, 1685153, 2491325,
        8238582, 3020393, 1753, 6715099, 1254190, 1716814, 4620952, 586241,
        4340221, 7277073, 3965306, 3033742, 2192938, 7325939, 635956, 1834526,
        1354892, 545376, 1780227, 1723229, 3747250, 6022044, 822541, 2033807,
        6201452, 860144, 3284915, 4148469, 3180456, 303005, 2678278, 6386371,
        2513018, 3994671, 2659525, 1163598, 5737437, 7987710, 6400920, 7852436,
        6458423, 553718, 7198174, 2647994, 1772588, 7598542, 2455377, 7872490,
        7070156, 8165537, 2772600, 3980599, 7140506, 3123762, 2454145, 1623354,
        2039144, 8240173, 6084020, 4022750, 5992904, 4405932, 3410568, 6987258,
        2998219, 1370517, 6444997, 6352379, 8368000, 5365997, 3956944, 7200804,
        3472069, 5274859, 636927, 338420, 6653329, 731434, 3551006, 7655613,
        7767179, 7609976, 2660408, 1615530, 2296099, 2193087, 27812, 6006015,
        818761, 3430436, 3716946, 2612853, 8111961, 4849188, 4611469, 6903432,
        89301, 8368538, 1455890, 5011144, 3195676, 5454363, 1596822, 2743411,
        458740, 508145, 8293209, 3009748, 4234153, 6480365, 8129971, 1187885,
        6161950, 3363542, 59148, 2569011, 3838479, 2185084, 1009365, 6275131,
        6500539, 1514152, 3956745, 749577, 3611750, 4606686, 6695264, 5889092,
        141835, 5360024, 8378664, 1665318, 7126227, 6663603, 3759465, 7794176,
        4040196, 1103344, 4415111, 5346675, 6187479, 1054478, 7744461, 6545891,
        7025525, 7835041, 6600190, 6657188, 4633167, 2358373, 7557876, 6346610,
        2178965, 7520273, 5095502, 4231948, 5199961, 8077412, 5702139, 1994046,
        5867399, 4385746, 5720892, 7216819, 2642980, 392707, 1979497, 527981};
    cudaMemcpyToSymbol(omega_pow_use_rank, omega_pow_use_rank_cpu, sizeof(omega_pow_use_rank_cpu));
    err = cudaGetLastError();
    printf("%s\n", cudaGetErrorString(err));
}

__global__ void NTT(int64 *a, int64 *out)
{
    int64 *vec_1 = a + blockIdx.x * 256;
    int64 *vec_2 = out + blockIdx.x * 256;
    int idx = threadIdx.x;
    int idx0 = idx;
    int idx1 = idx + 32;
    int idx2 = idx + 64;
    int idx3 = idx + 96;
    int idx4 = idx + 128;
    int idx5 = idx + 160;
    int idx6 = idx + 192;
    int idx7 = idx + 224;

    vec_2[idx0] = vec_1[bit_reverse[idx0]];
    vec_2[idx1] = vec_1[bit_reverse[idx1]];
    vec_2[idx2] = vec_1[bit_reverse[idx2]];
    vec_2[idx3] = vec_1[bit_reverse[idx3]];
    vec_2[idx4] = vec_1[bit_reverse[idx4]];
    vec_2[idx5] = vec_1[bit_reverse[idx5]];
    vec_2[idx6] = vec_1[bit_reverse[idx6]];
    vec_2[idx7] = vec_1[bit_reverse[idx7]];

    vec_1[idx0] = (vec_2[idx0 & 0xFFFFFFFE] + (int64)omega_pow_use_rank[(idx0 & 0x01) | 0x02] * vec_2[idx0 | 0x00000001]) % 8380417;
    vec_1[idx1] = (vec_2[idx1 & 0xFFFFFFFE] + (int64)omega_pow_use_rank[(idx1 & 0x01) | 0x02] * vec_2[idx1 | 0x00000001]) % 8380417;
    vec_1[idx2] = (vec_2[idx2 & 0xFFFFFFFE] + (int64)omega_pow_use_rank[(idx2 & 0x01) | 0x02] * vec_2[idx2 | 0x00000001]) % 8380417;
    vec_1[idx3] = (vec_2[idx3 & 0xFFFFFFFE] + (int64)omega_pow_use_rank[(idx3 & 0x01) | 0x02] * vec_2[idx3 | 0x00000001]) % 8380417;
    vec_1[idx4] = (vec_2[idx4 & 0xFFFFFFFE] + (int64)omega_pow_use_rank[(idx4 & 0x01) | 0x02] * vec_2[idx4 | 0x00000001]) % 8380417;
    vec_1[idx5] = (vec_2[idx5 & 0xFFFFFFFE] + (int64)omega_pow_use_rank[(idx5 & 0x01) | 0x02] * vec_2[idx5 | 0x00000001]) % 8380417;
    vec_1[idx6] = (vec_2[idx6 & 0xFFFFFFFE] + (int64)omega_pow_use_rank[(idx6 & 0x01) | 0x02] * vec_2[idx6 | 0x00000001]) % 8380417;
    vec_1[idx7] = (vec_2[idx7 & 0xFFFFFFFE] + (int64)omega_pow_use_rank[(idx7 & 0x01) | 0x02] * vec_2[idx7 | 0x00000001]) % 8380417;

    vec_2[idx0] = (vec_1[idx0 & 0xFFFFFFFD] + (int64)omega_pow_use_rank[(idx0 & 0x03) | 0x04] * vec_1[idx0 | 0x00000002]) % 8380417;
    vec_2[idx1] = (vec_1[idx1 & 0xFFFFFFFD] + (int64)omega_pow_use_rank[(idx1 & 0x03) | 0x04] * vec_1[idx1 | 0x00000002]) % 8380417;
    vec_2[idx2] = (vec_1[idx2 & 0xFFFFFFFD] + (int64)omega_pow_use_rank[(idx2 & 0x03) | 0x04] * vec_1[idx2 | 0x00000002]) % 8380417;
    vec_2[idx3] = (vec_1[idx3 & 0xFFFFFFFD] + (int64)omega_pow_use_rank[(idx3 & 0x03) | 0x04] * vec_1[idx3 | 0x00000002]) % 8380417;
    vec_2[idx4] = (vec_1[idx4 & 0xFFFFFFFD] + (int64)omega_pow_use_rank[(idx4 & 0x03) | 0x04] * vec_1[idx4 | 0x00000002]) % 8380417;
    vec_2[idx5] = (vec_1[idx5 & 0xFFFFFFFD] + (int64)omega_pow_use_rank[(idx5 & 0x03) | 0x04] * vec_1[idx5 | 0x00000002]) % 8380417;
    vec_2[idx6] = (vec_1[idx6 & 0xFFFFFFFD] + (int64)omega_pow_use_rank[(idx6 & 0x03) | 0x04] * vec_1[idx6 | 0x00000002]) % 8380417;
    vec_2[idx7] = (vec_1[idx7 & 0xFFFFFFFD] + (int64)omega_pow_use_rank[(idx7 & 0x03) | 0x04] * vec_1[idx7 | 0x00000002]) % 8380417;

    vec_1[idx0] = (vec_2[idx0 & 0xFFFFFFFB] + (int64)omega_pow_use_rank[(idx0 & 0x07) | 0x08] * vec_2[idx0 | 0x00000004]) % 8380417;
    vec_1[idx1] = (vec_2[idx1 & 0xFFFFFFFB] + (int64)omega_pow_use_rank[(idx1 & 0x07) | 0x08] * vec_2[idx1 | 0x00000004]) % 8380417;
    vec_1[idx2] = (vec_2[idx2 & 0xFFFFFFFB] + (int64)omega_pow_use_rank[(idx2 & 0x07) | 0x08] * vec_2[idx2 | 0x00000004]) % 8380417;
    vec_1[idx3] = (vec_2[idx3 & 0xFFFFFFFB] + (int64)omega_pow_use_rank[(idx3 & 0x07) | 0x08] * vec_2[idx3 | 0x00000004]) % 8380417;
    vec_1[idx4] = (vec_2[idx4 & 0xFFFFFFFB] + (int64)omega_pow_use_rank[(idx4 & 0x07) | 0x08] * vec_2[idx4 | 0x00000004]) % 8380417;
    vec_1[idx5] = (vec_2[idx5 & 0xFFFFFFFB] + (int64)omega_pow_use_rank[(idx5 & 0x07) | 0x08] * vec_2[idx5 | 0x00000004]) % 8380417;
    vec_1[idx6] = (vec_2[idx6 & 0xFFFFFFFB] + (int64)omega_pow_use_rank[(idx6 & 0x07) | 0x08] * vec_2[idx6 | 0x00000004]) % 8380417;
    vec_1[idx7] = (vec_2[idx7 & 0xFFFFFFFB] + (int64)omega_pow_use_rank[(idx7 & 0x07) | 0x08] * vec_2[idx7 | 0x00000004]) % 8380417;

    vec_2[idx0] = (vec_1[idx0 & 0xFFFFFFF7] + (int64)omega_pow_use_rank[(idx0 & 0x0F) | 0x10] * vec_1[idx0 | 0x00000008]) % 8380417;
    vec_2[idx1] = (vec_1[idx1 & 0xFFFFFFF7] + (int64)omega_pow_use_rank[(idx1 & 0x0F) | 0x10] * vec_1[idx1 | 0x00000008]) % 8380417;
    vec_2[idx2] = (vec_1[idx2 & 0xFFFFFFF7] + (int64)omega_pow_use_rank[(idx2 & 0x0F) | 0x10] * vec_1[idx2 | 0x00000008]) % 8380417;
    vec_2[idx3] = (vec_1[idx3 & 0xFFFFFFF7] + (int64)omega_pow_use_rank[(idx3 & 0x0F) | 0x10] * vec_1[idx3 | 0x00000008]) % 8380417;
    vec_2[idx4] = (vec_1[idx4 & 0xFFFFFFF7] + (int64)omega_pow_use_rank[(idx4 & 0x0F) | 0x10] * vec_1[idx4 | 0x00000008]) % 8380417;
    vec_2[idx5] = (vec_1[idx5 & 0xFFFFFFF7] + (int64)omega_pow_use_rank[(idx5 & 0x0F) | 0x10] * vec_1[idx5 | 0x00000008]) % 8380417;
    vec_2[idx6] = (vec_1[idx6 & 0xFFFFFFF7] + (int64)omega_pow_use_rank[(idx6 & 0x0F) | 0x10] * vec_1[idx6 | 0x00000008]) % 8380417;
    vec_2[idx7] = (vec_1[idx7 & 0xFFFFFFF7] + (int64)omega_pow_use_rank[(idx7 & 0x0F) | 0x10] * vec_1[idx7 | 0x00000008]) % 8380417;

    vec_1[idx0] = (vec_2[idx0 & 0xFFFFFFEF] + (int64)omega_pow_use_rank[(idx0 & 0x1F) | 0x20] * vec_2[idx0 | 0x00000010]) % 8380417;
    vec_1[idx1] = (vec_2[idx1 & 0xFFFFFFEF] + (int64)omega_pow_use_rank[(idx1 & 0x1F) | 0x20] * vec_2[idx1 | 0x00000010]) % 8380417;
    vec_1[idx2] = (vec_2[idx2 & 0xFFFFFFEF] + (int64)omega_pow_use_rank[(idx2 & 0x1F) | 0x20] * vec_2[idx2 | 0x00000010]) % 8380417;
    vec_1[idx3] = (vec_2[idx3 & 0xFFFFFFEF] + (int64)omega_pow_use_rank[(idx3 & 0x1F) | 0x20] * vec_2[idx3 | 0x00000010]) % 8380417;
    vec_1[idx4] = (vec_2[idx4 & 0xFFFFFFEF] + (int64)omega_pow_use_rank[(idx4 & 0x1F) | 0x20] * vec_2[idx4 | 0x00000010]) % 8380417;
    vec_1[idx5] = (vec_2[idx5 & 0xFFFFFFEF] + (int64)omega_pow_use_rank[(idx5 & 0x1F) | 0x20] * vec_2[idx5 | 0x00000010]) % 8380417;
    vec_1[idx6] = (vec_2[idx6 & 0xFFFFFFEF] + (int64)omega_pow_use_rank[(idx6 & 0x1F) | 0x20] * vec_2[idx6 | 0x00000010]) % 8380417;
    vec_1[idx7] = (vec_2[idx7 & 0xFFFFFFEF] + (int64)omega_pow_use_rank[(idx7 & 0x1F) | 0x20] * vec_2[idx7 | 0x00000010]) % 8380417;

    vec_2[idx0] = (vec_1[idx0 & 0xFFFFFFDF] + (int64)omega_pow_use_rank[(idx0 & 0x3F) | 0x40] * vec_1[idx0 | 0x00000020]) % 8380417;
    vec_2[idx1] = (vec_1[idx1 & 0xFFFFFFDF] + (int64)omega_pow_use_rank[(idx1 & 0x3F) | 0x40] * vec_1[idx1 | 0x00000020]) % 8380417;
    vec_2[idx2] = (vec_1[idx2 & 0xFFFFFFDF] + (int64)omega_pow_use_rank[(idx2 & 0x3F) | 0x40] * vec_1[idx2 | 0x00000020]) % 8380417;
    vec_2[idx3] = (vec_1[idx3 & 0xFFFFFFDF] + (int64)omega_pow_use_rank[(idx3 & 0x3F) | 0x40] * vec_1[idx3 | 0x00000020]) % 8380417;
    vec_2[idx4] = (vec_1[idx4 & 0xFFFFFFDF] + (int64)omega_pow_use_rank[(idx4 & 0x3F) | 0x40] * vec_1[idx4 | 0x00000020]) % 8380417;
    vec_2[idx5] = (vec_1[idx5 & 0xFFFFFFDF] + (int64)omega_pow_use_rank[(idx5 & 0x3F) | 0x40] * vec_1[idx5 | 0x00000020]) % 8380417;
    vec_2[idx6] = (vec_1[idx6 & 0xFFFFFFDF] + (int64)omega_pow_use_rank[(idx6 & 0x3F) | 0x40] * vec_1[idx6 | 0x00000020]) % 8380417;
    vec_2[idx7] = (vec_1[idx7 & 0xFFFFFFDF] + (int64)omega_pow_use_rank[(idx7 & 0x3F) | 0x40] * vec_1[idx7 | 0x00000020]) % 8380417;

    vec_1[idx0] = (vec_2[idx0 & 0xFFFFFFBF] + (int64)omega_pow_use_rank[(idx0 & 0x7F) | 0x80] * vec_2[idx0 | 0x00000040]) % 8380417;
    vec_1[idx1] = (vec_2[idx1 & 0xFFFFFFBF] + (int64)omega_pow_use_rank[(idx1 & 0x7F) | 0x80] * vec_2[idx1 | 0x00000040]) % 8380417;
    vec_1[idx2] = (vec_2[idx2 & 0xFFFFFFBF] + (int64)omega_pow_use_rank[(idx2 & 0x7F) | 0x80] * vec_2[idx2 | 0x00000040]) % 8380417;
    vec_1[idx3] = (vec_2[idx3 & 0xFFFFFFBF] + (int64)omega_pow_use_rank[(idx3 & 0x7F) | 0x80] * vec_2[idx3 | 0x00000040]) % 8380417;
    vec_1[idx4] = (vec_2[idx4 & 0xFFFFFFBF] + (int64)omega_pow_use_rank[(idx4 & 0x7F) | 0x80] * vec_2[idx4 | 0x00000040]) % 8380417;
    vec_1[idx5] = (vec_2[idx5 & 0xFFFFFFBF] + (int64)omega_pow_use_rank[(idx5 & 0x7F) | 0x80] * vec_2[idx5 | 0x00000040]) % 8380417;
    vec_1[idx6] = (vec_2[idx6 & 0xFFFFFFBF] + (int64)omega_pow_use_rank[(idx6 & 0x7F) | 0x80] * vec_2[idx6 | 0x00000040]) % 8380417;
    vec_1[idx7] = (vec_2[idx7 & 0xFFFFFFBF] + (int64)omega_pow_use_rank[(idx7 & 0x7F) | 0x80] * vec_2[idx7 | 0x00000040]) % 8380417;

    vec_2[idx0] = (vec_1[idx0 & 0xFFFFFF7F] + (int64)omega_pow_use_rank[idx0 | 0x100] * vec_1[idx0 | 0x00000080]) % 8380417;
    vec_2[idx1] = (vec_1[idx1 & 0xFFFFFF7F] + (int64)omega_pow_use_rank[idx1 | 0x100] * vec_1[idx1 | 0x00000080]) % 8380417;
    vec_2[idx2] = (vec_1[idx2 & 0xFFFFFF7F] + (int64)omega_pow_use_rank[idx2 | 0x100] * vec_1[idx2 | 0x00000080]) % 8380417;
    vec_2[idx3] = (vec_1[idx3 & 0xFFFFFF7F] + (int64)omega_pow_use_rank[idx3 | 0x100] * vec_1[idx3 | 0x00000080]) % 8380417;
    vec_2[idx4] = (vec_1[idx4 & 0xFFFFFF7F] + (int64)omega_pow_use_rank[idx4 | 0x100] * vec_1[idx4 | 0x00000080]) % 8380417;
    vec_2[idx5] = (vec_1[idx5 & 0xFFFFFF7F] + (int64)omega_pow_use_rank[idx5 | 0x100] * vec_1[idx5 | 0x00000080]) % 8380417;
    vec_2[idx6] = (vec_1[idx6 & 0xFFFFFF7F] + (int64)omega_pow_use_rank[idx6 | 0x100] * vec_1[idx6 | 0x00000080]) % 8380417;
    vec_2[idx7] = (vec_1[idx7 & 0xFFFFFF7F] + (int64)omega_pow_use_rank[idx7 | 0x100] * vec_1[idx7 | 0x00000080]) % 8380417;
}