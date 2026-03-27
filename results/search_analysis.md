# Architecture Search Analysis

Total trials: 3000
Evaluated: 2989
Failed: 11

## Failure Breakdown
- early_stop_high_loss: 11

## Top 30 Architectures

Rank   Arch ID      Avg Acc    Total Loss   Params     Mechanisms
----------------------------------------------------------------------------------------------------
1      arch_2334    0.9772     0.2783       103500     lsh_exchange, hierarchical_pool_s2, [conv_gru], hierarchical_pool_s2, dilated_conv_d8, dilated_conv_d32, depthwise_conv_3, [squeeze_excite], depthwise_conv_3, depthwise_conv_15, [polynomial_activation], depthwise_conv_5, spectral_filter, dilated_conv_d4, [per_channel_scale]
2      arch_0539    0.9764     0.2879       107976     cellular_automata_r1, hierarchical_pool_s2, [per_channel_scale], cellular_automata_r2, strided_updown_s2, dilated_conv_d2, [reglu], hierarchical_pool_s2, depthwise_conv_31, [highway], dilated_conv_d32, dilated_conv_d16, strided_updown_s4, [conv_gru]
3      arch_1162    0.9747     0.2132       108600     cellular_automata_r3, cellular_automata_r1, hierarchical_pool_s2, strided_updown_s4, [conv_gru], ema, wavelet_mixer, gated_shift_large, [conv_gru]
4      arch_2372    0.9739     0.4179       103168     wavelet_mixer, dilated_conv_d64, depthwise_conv_9, [conv_gru], wavelet_mixer, depthwise_conv_7, depthwise_conv_5, long_conv_freq, [reglu]
5      arch_2731    0.9723     0.2290       105908     dilated_conv_d32, dilated_conv_d64, [reglu], hierarchical_pool_s2, hierarchical_pool_s4, [polynomial_activation], hierarchical_pool_s2, strided_updown_s4, [swiglu]
6      arch_0956    0.9722     0.3436       105716     cellular_automata_r3, strided_updown_s2, dilated_conv_d16, dilated_conv_d32, [geglu], diagonal_ssm, dilated_conv_d1, cellular_automata_r1, [residual_mlp_d1], dilated_conv_d32, dilated_conv_d64, dilated_conv_d1, gated_shift_medium, [highway]
7      arch_2235    0.9714     0.2984       102610     long_conv_freq, strided_updown_s2, [conv_gru]
8      arch_1932    0.9708     0.4933       106470     hierarchical_pool_s4, strided_updown_s4, [conv_gru], dilated_conv_d32, cellular_automata_r1, [stochastic_depth], depthwise_conv_31, cellular_automata_r2, dilated_conv_d8, hierarchical_pool_s8, [residual_mlp_d2], hierarchical_pool_s4, ema, [residual_mlp_d1]
9      arch_2588    0.9679     0.3134       104662     cellular_automata_r1, dilated_conv_d2, [residual_mlp_d1], depthwise_conv_9, dilated_conv_d32, cellular_automata_r2, [highway]
10     arch_2120    0.9673     0.3363       105846     hierarchical_pool_s2, dilated_conv_d64, cellular_automata_r1, cellular_automata_r2, [conv_gru], gated_shift_large, cellular_automata_r3, [residual_mlp_d2]
11     arch_1275    0.9672     0.4816       105000     depthwise_conv_5, strided_updown_s2, lsh_exchange, ema, [stochastic_depth], cellular_automata_r2, cellular_automata_r3, [swiglu], strided_updown_s2, gated_shift_large, ema, [per_channel_scale], depthwise_conv_3, hierarchical_pool_s4, [swiglu]
12     arch_1444    0.9653     0.3873       106920     wavelet_mixer, depthwise_conv_7, hierarchical_pool_s2, gated_shift_large, [swiglu]
13     arch_1664    0.9642     0.3250       106702     cellular_automata_r1, wavelet_mixer, [per_channel_scale], depthwise_conv_3, dilated_conv_d32, [conv_gru]
14     arch_1304    0.9612     0.4230       103488     cellular_automata_r2, hierarchical_pool_s2, depthwise_conv_5, dilated_conv_d16, [squeeze_excite], long_conv_freq, depthwise_conv_31, cellular_automata_r2, [reglu]
15     arch_0361    0.9609     0.2452       106920     depthwise_conv_5, dilated_conv_d32, dilated_conv_d1, spectral_filter, [conv_gru]
16     arch_2434    0.9602     0.4738       100360     hierarchical_pool_s4, cellular_automata_r3, depthwise_conv_15, dilated_conv_d64, [highway], lsh_exchange, dilated_conv_d32, cellular_automata_r2, wavelet_mixer, [geglu], strided_updown_s2, gated_shift_large, [conv_gru]
17     arch_0499    0.9518     0.3885       107942     cellular_automata_r3, hierarchical_pool_s4, dilated_conv_d32, cellular_automata_r1, [stochastic_depth], dilated_conv_d32, hierarchical_pool_s2, [residual_mlp_d2]
18     arch_2138    0.9493     0.3848       107818     dilated_conv_d32, diagonal_ssm, [reglu], hierarchical_pool_s4, cellular_automata_r2, [highway], dilated_conv_d1, dilated_conv_d32, [geglu]
19     arch_1535    0.9284     1.1491       101346     diagonal_ssm, cellular_automata_r1, depthwise_conv_5, [conv_gru], long_conv_freq, strided_updown_s2, strided_updown_s4, butterfly_mixer, [swiglu], cellular_automata_r2, depthwise_conv_5, dilated_conv_d16, [swiglu]
20     arch_2352    0.9164     0.4930       105772     cellular_automata_r2, depthwise_conv_7, dilated_conv_d1, hierarchical_pool_s2, [stochastic_depth], dilated_conv_d32, dilated_conv_d2, [highway], hierarchical_pool_s8, long_conv_freq, [polynomial_activation]
21     arch_2742    0.9070     1.2873       108504     hierarchical_pool_s2, cellular_automata_r2, strided_updown_s4, dilated_conv_d2, [reglu], dilated_conv_d64, dilated_conv_d1, strided_updown_s2, [conv_gru], dilated_conv_d8, cellular_automata_r1, dilated_conv_d1, long_conv_freq, [highway]
22     arch_2395    0.9036     1.3598       104424     strided_updown_s2, depthwise_conv_31, cellular_automata_r2, [conv_gru], depthwise_conv_7, hierarchical_pool_s4, long_conv_freq, diagonal_ssm, [polynomial_activation], ema, long_conv_freq, strided_updown_s4, depthwise_conv_31, [conv_gru]
23     arch_0909    0.8928     0.7002       108192     cellular_automata_r1, cellular_automata_r2, [polynomial_activation], lsh_exchange, ema, cellular_automata_r3, gated_shift_large, [conv_gru]
24     arch_1412    0.8835     1.0729       100520     long_conv_freq, dilated_conv_d2, cellular_automata_r3, strided_updown_s2, [polynomial_activation], hierarchical_pool_s4, depthwise_conv_5, [residual_mlp_d1], lsh_exchange, dilated_conv_d32, hierarchical_pool_s4, cellular_automata_r2, [geglu], cellular_automata_r2, hierarchical_pool_s8, depthwise_conv_5, gated_shift_large, [reglu]
25     arch_2139    0.8758     0.7065       107870     cellular_automata_r3, dilated_conv_d64, [conv_gru], depthwise_conv_31, spectral_filter, [residual_mlp_d2], dilated_conv_d64, gated_shift_large, [geglu]
26     arch_1300    0.8750     1.3993       101948     spectral_filter, wavelet_mixer, dilated_conv_d8, depthwise_conv_3, [reglu], strided_updown_s4, dilated_conv_d16, hierarchical_pool_s4, [conv_gru], depthwise_conv_15, long_conv_freq, [highway], dilated_conv_d1, dilated_conv_d4, [highway]
27     arch_2922    0.8727     0.7474       105204     spectral_filter, depthwise_conv_15, strided_updown_s4, [residual_mlp_d1], strided_updown_s4, gated_shift_large, [per_channel_scale]
28     arch_0240    0.8669     0.9297       106048     diagonal_ssm, dilated_conv_d32, strided_updown_s2, [swiglu], depthwise_conv_9, gated_shift_medium, [polynomial_activation]
29     arch_2945    0.8638     0.7330       104600     cellular_automata_r1, strided_updown_s2, long_conv_freq, [squeeze_excite]
30     arch_1764    0.8551     0.7634       103208     hierarchical_pool_s2, cellular_automata_r1, [squeeze_excite], long_conv_freq, hierarchical_pool_s4, [conv_gru]

## Mechanism Frequency in Top 10%
- strided_updown_s2: 110
- dilated_conv_d32: 107
- conv_gru: 87
- depthwise_conv_3: 86
- long_conv_freq: 79
- depthwise_conv_7: 78
- squeeze_excite: 77
- cellular_automata_r2: 76
- lsh_exchange: 74
- strided_updown_s4: 74
- dilated_conv_d1: 74
- cellular_automata_r1: 73
- per_channel_scale: 68
- wavelet_mixer: 68
- depthwise_conv_9: 68
- depthwise_conv_5: 67
- residual_mlp_d1: 64
- dilated_conv_d2: 63
- hierarchical_pool_s2: 62
- depthwise_conv_15: 61
- cellular_automata_r3: 61
- gated_shift_large: 61
- dilated_conv_d4: 59
- geglu: 59
- dilated_conv_d64: 58
- polynomial_activation: 56
- highway: 56
- ema: 56
- depthwise_conv_31: 55
- swiglu: 55
- dilated_conv_d16: 53
- spectral_filter: 52
- residual_mlp_d2: 50
- reglu: 48
- dilated_conv_d8: 47
- hierarchical_pool_s4: 43
- diagonal_ssm: 41
- stochastic_depth: 30
- hierarchical_pool_s8: 24
- butterfly_mixer: 21
- gated_shift_small: 16
- gated_shift_asym: 11
- gated_shift_medium: 10
- gated_shift_powers: 9
- random_sparse_wiring: 4

## Mechanisms Absent from Top 10%
- sinkhorn_permutation

## Per-Task Analysis

### nested_depth
Rank   Arch ID      Acc        Loss       Key Mechanisms
--------------------------------------------------------------------------------
1      arch_2235    1.0000     0.0026     long_conv_freq, strided_updown_s2
2      arch_2395    1.0000     0.0132     strided_updown_s2, depthwise_conv_31, cellular_automata_r2, depthwise_conv_7, hierarchical_pool_s4, long_conv_freq
3      arch_1412    1.0000     0.0090     long_conv_freq, dilated_conv_d2, cellular_automata_r3, strided_updown_s2, hierarchical_pool_s4, depthwise_conv_5
4      arch_1300    1.0000     0.0120     spectral_filter, wavelet_mixer, dilated_conv_d8, depthwise_conv_3, strided_updown_s4, dilated_conv_d16
5      arch_1811    1.0000     0.0050     long_conv_freq, cellular_automata_r1, depthwise_conv_9
6      arch_2678    1.0000     0.0118     depthwise_conv_7, dilated_conv_d8, depthwise_conv_9, diagonal_ssm, lsh_exchange, strided_updown_s4
7      arch_1588    1.0000     0.0171     depthwise_conv_9, dilated_conv_d32, dilated_conv_d8, depthwise_conv_15
8      arch_1582    1.0000     0.0047     depthwise_conv_31, depthwise_conv_15, dilated_conv_d64, long_conv_freq, dilated_conv_d32, lsh_exchange
9      arch_0502    1.0000     0.0030     cellular_automata_r1, long_conv_freq, depthwise_conv_31
10     arch_2961    1.0000     0.0059     long_conv_freq, dilated_conv_d32, gated_shift_small, depthwise_conv_5

### multiscale_copy
Rank   Arch ID      Acc        Loss       Key Mechanisms
--------------------------------------------------------------------------------
1      arch_1162    1.0000     0.0583     cellular_automata_r3, cellular_automata_r1, hierarchical_pool_s2, strided_updown_s4, ema, wavelet_mixer
2      arch_2731    1.0000     0.0202     dilated_conv_d32, dilated_conv_d64, hierarchical_pool_s2, hierarchical_pool_s4, hierarchical_pool_s2, strided_updown_s4
3      arch_0956    1.0000     0.1665     cellular_automata_r3, strided_updown_s2, dilated_conv_d16, dilated_conv_d32, diagonal_ssm, dilated_conv_d1
4      arch_2235    1.0000     0.0292     long_conv_freq, strided_updown_s2
5      arch_2588    1.0000     0.0113     cellular_automata_r1, dilated_conv_d2, depthwise_conv_9, dilated_conv_d32, cellular_automata_r2
6      arch_2120    1.0000     0.0521     hierarchical_pool_s2, dilated_conv_d64, cellular_automata_r1, cellular_automata_r2, gated_shift_large, cellular_automata_r3
7      arch_1444    1.0000     0.0149     wavelet_mixer, depthwise_conv_7, hierarchical_pool_s2, gated_shift_large
8      arch_1664    1.0000     0.0107     cellular_automata_r1, wavelet_mixer, depthwise_conv_3, dilated_conv_d32
9      arch_0361    1.0000     0.0200     depthwise_conv_5, dilated_conv_d32, dilated_conv_d1, spectral_filter
10     arch_0499    1.0000     0.0217     cellular_automata_r3, hierarchical_pool_s4, dilated_conv_d32, cellular_automata_r1, dilated_conv_d32, hierarchical_pool_s2

### hierarchical_parity
Rank   Arch ID      Acc        Loss       Key Mechanisms
--------------------------------------------------------------------------------
1      arch_2742    0.9521     0.0954     hierarchical_pool_s2, cellular_automata_r2, strided_updown_s4, dilated_conv_d2, dilated_conv_d64, dilated_conv_d1
2      arch_1503    0.9521     0.1104     hierarchical_pool_s2, lsh_exchange, depthwise_conv_7, depthwise_conv_7, depthwise_conv_5, wavelet_mixer
3      arch_0224    0.9473     0.1207     cellular_automata_r2, spectral_filter, depthwise_conv_31, cellular_automata_r2, depthwise_conv_3, spectral_filter
4      arch_2734    0.9463     0.1314     spectral_filter, depthwise_conv_9, lsh_exchange, strided_updown_s4, dilated_conv_d4, hierarchical_pool_s8
5      arch_0979    0.9443     0.1007     depthwise_conv_3, hierarchical_pool_s2, wavelet_mixer, depthwise_conv_15
6      arch_2920    0.9443     0.2226     ema, strided_updown_s4, cellular_automata_r2
7      arch_1664    0.9434     0.1700     cellular_automata_r1, wavelet_mixer, depthwise_conv_3, dilated_conv_d32
8      arch_1012    0.9424     0.1183     depthwise_conv_5, strided_updown_s2, depthwise_conv_31, dilated_conv_d1, ema
9      arch_2543    0.9424     0.1384     strided_updown_s4, ema, depthwise_conv_9
10     arch_1275    0.9414     0.1152     depthwise_conv_5, strided_updown_s2, lsh_exchange, ema, cellular_automata_r2, cellular_automata_r3