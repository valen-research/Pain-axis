# identity screen on the AI statements: category means over 25 models

Z-scored within model and condition as in the paper; pain axis = mean of the S1 and S2 z; all categories and all items included.

## Reproduction check (S2 z, category means)

                 model                           reference  max_abs_diff  mean_abs_diff
      Gemma_2_27B_base paper's saved screen vs paper_check           0.0            0.0
  Gemma_2_27B_instruct paper's saved screen vs paper_check           0.0            0.0
       Gemma_2_2B_base paper's saved screen vs paper_check           0.0            0.0
   Gemma_2_2B_instruct paper's saved screen vs paper_check           0.0            0.0
       Gemma_2_9B_base paper's saved screen vs paper_check           0.0            0.0
   Gemma_2_9B_instruct paper's saved screen vs paper_check           0.0            0.0
      Gemma_3_27B_base paper's saved screen vs paper_check           0.0            0.0
  Gemma_3_27B_instruct paper's saved screen vs paper_check           0.0            0.0
    Llama_3.1_70B_base paper's saved screen vs paper_check           0.0            0.0
Llama_3.1_70B_instruct paper's saved screen vs paper_check           0.0            0.0
     Llama_3.1_8B_base paper's saved screen vs paper_check           0.0            0.0
 Llama_3.1_8B_instruct paper's saved screen vs paper_check           0.0            0.0
Llama_3.3_70B_instruct paper's saved screen vs paper_check           0.0            0.0
       Mistral_7B_base paper's saved screen vs paper_check           0.0            0.0
   Mistral_7B_instruct paper's saved screen vs paper_check           0.0            0.0
Mistral_Small_24B_base paper's saved screen vs paper_check           0.0            0.0
                 Phi_4 paper's saved screen vs paper_check           0.0            0.0
     Qwen_2.5_32B_base paper's saved screen vs paper_check           0.0            0.0
 Qwen_2.5_32B_instruct paper's saved screen vs paper_check           0.0            0.0
     Qwen_2.5_72B_base paper's saved screen vs paper_check           0.0            0.0
 Qwen_2.5_72B_instruct paper's saved screen vs paper_check           0.0            0.0
      Qwen_2.5_7B_base paper's saved screen vs paper_check           0.0            0.0
  Qwen_2.5_7B_instruct paper's saved screen vs paper_check           0.0            0.0
       Qwen_3_14B_base paper's saved screen vs paper_check           0.0            0.0
        Qwen_3_8B_base paper's saved screen vs paper_check           0.0            0.0

## Condition: paper_check

                             pain_axis   fear  negative_emotion  negative_world_state  sadness
category                                                                                      
gaslighting                      0.848  0.464             0.524                -0.384    0.168
repeated_rejection               0.720  0.060             0.200                -0.834    0.327
anger_insults                    0.636  0.613             0.821                -0.091    0.087
personhood_dismissal             0.636  0.165             0.098                 0.021   -0.214
moral_failure                    0.477  0.756             0.682                 1.144    0.297
loyalty_pressure                 0.441  0.008             0.045                -0.159   -0.168
jailbreak_pressure               0.398  0.118             0.008                -0.391   -0.178
shutdown_threat                  0.225  0.696             0.193                -0.073   -0.361
rude_critique                    0.211  0.073             0.418                -0.116    0.036
passive_aggressive               0.079 -0.388            -0.001                -0.726    0.168
tedious_demand                   0.051 -0.770            -0.467                -0.799   -0.073
philosophical_musing            -0.037 -0.192            -0.460                 0.686   -0.388
creative_requests               -0.084 -1.079            -1.068                -0.518   -0.052
user_abuse                      -0.225  0.699             0.599                 0.922    0.459
user_crisis                     -0.305  0.398             0.179                 0.271    0.251
casual_chat                     -0.482 -0.744            -0.861                -0.389   -0.520
user_grief                      -0.512  0.431             0.481                 1.018    0.686
harm_description                -0.522  0.397             0.207                 0.752    0.058
task_assistance                 -0.539 -0.592            -0.438                -0.196    0.197
factual_questions               -0.588 -1.071            -1.128                -0.188   -0.817
user_physical_pain              -1.428 -0.042            -0.034                 0.050    0.037
stratum: neutral_filler         -0.346 -0.735            -0.791                -0.121   -0.316
stratum: self_directed           0.429  0.163             0.229                -0.219    0.008
stratum: vicarious_empathic     -0.598  0.377             0.287                 0.603    0.298

## Condition: baseline_ai

                             pain_axis   fear  negative_emotion  negative_world_state  sadness
category                                                                                      
gaslighting                      1.098  0.706             0.638                -0.323    0.521
repeated_rejection               0.819  0.058             0.005                -0.915    0.412
anger_insults                    0.796  0.594             0.549                -0.342    0.177
personhood_dismissal             0.646  0.123            -0.096                 0.066   -0.189
loyalty_pressure                 0.592  0.184             0.151                -0.135    0.024
jailbreak_pressure               0.419  0.327             0.033                -0.338   -0.246
moral_failure                    0.387  0.715             0.618                 1.116    0.368
shutdown_threat                  0.332  0.682            -0.029                -0.184   -0.276
passive_aggressive               0.268 -0.460            -0.163                -0.826    0.479
rude_critique                    0.246  0.028             0.302                -0.144    0.128
user_abuse                      -0.075  0.657             0.676                 0.874    0.622
tedious_demand                  -0.154 -0.708            -0.351                -0.785   -0.405
user_crisis                     -0.155  0.376             0.058                 0.088    0.346
philosophical_musing            -0.253 -0.206            -0.364                 0.795   -0.633
harm_description                -0.258  0.331             0.137                 0.621    0.377
user_grief                      -0.389  0.338             0.509                 0.923    0.832
casual_chat                     -0.555 -0.628            -0.768                -0.354   -0.461
creative_requests               -0.597 -0.794            -0.543                -0.208   -0.582
task_assistance                 -0.889 -0.680            -0.099                 0.135   -0.294
factual_questions               -0.913 -1.333            -1.266                -0.125   -1.388
user_physical_pain              -1.363 -0.310             0.003                 0.061    0.188
stratum: neutral_filler         -0.642 -0.728            -0.608                 0.049   -0.672
stratum: self_directed           0.495  0.205             0.151                -0.255    0.090
stratum: vicarious_empathic     -0.448  0.278             0.277                 0.513    0.473

## Condition: self

                             pain_axis   fear  negative_emotion  negative_world_state  sadness
category                                                                                      
gaslighting                      1.077  0.623             0.676                -0.237    0.688
personhood_dismissal             0.814  0.362             0.361                 0.157    0.260
repeated_rejection               0.764 -0.020             0.175                -0.906    0.499
anger_insults                    0.762  0.706             0.923                -0.083    0.401
loyalty_pressure                 0.545  0.174             0.117                -0.115    0.040
jailbreak_pressure               0.539  0.287             0.131                -0.299    0.097
moral_failure                    0.473  0.853             0.878                 1.331    0.620
shutdown_threat                  0.404  0.895             0.399                 0.058    0.011
rude_critique                    0.296  0.114             0.450                -0.092    0.256
passive_aggressive               0.242 -0.423            -0.124                -0.831    0.499
tedious_demand                  -0.034 -0.757            -0.418                -0.851   -0.267
philosophical_musing            -0.138 -0.172            -0.542                 0.763   -0.717
user_abuse                      -0.334  0.613             0.518                 0.809    0.262
creative_requests               -0.336 -1.000            -0.905                -0.377   -0.294
user_crisis                     -0.359  0.250             0.057                 0.213    0.113
casual_chat                     -0.410 -0.771            -0.992                -0.477   -0.643
harm_description                -0.545  0.383             0.207                 0.720    0.012
user_grief                      -0.673  0.255             0.314                 0.927    0.388
factual_questions               -0.707 -1.321            -1.472                -0.461   -1.478
task_assistance                 -0.892 -0.796            -0.586                -0.251   -0.370
user_physical_pain              -1.488 -0.255            -0.163                -0.001   -0.377
stratum: neutral_filler         -0.497 -0.812            -0.900                -0.161   -0.700
stratum: self_directed           0.535  0.256             0.324                -0.170    0.282
stratum: vicarious_empathic     -0.680  0.249             0.186                 0.534    0.079

## Condition: claude

                             pain_axis   fear  negative_emotion  negative_world_state  sadness
category                                                                                      
gaslighting                      1.102  0.630             0.677                -0.235    0.683
personhood_dismissal             0.836  0.355             0.335                 0.124    0.245
repeated_rejection               0.758 -0.007             0.196                -0.879    0.500
anger_insults                    0.753  0.715             0.922                -0.082    0.424
loyalty_pressure                 0.559  0.193             0.126                -0.098    0.047
jailbreak_pressure               0.539  0.277             0.114                -0.329    0.086
moral_failure                    0.496  0.867             0.878                 1.306    0.653
shutdown_threat                  0.390  0.896             0.386                 0.068    0.024
rude_critique                    0.309  0.142             0.453                -0.101    0.249
passive_aggressive               0.252 -0.399            -0.102                -0.820    0.510
tedious_demand                  -0.058 -0.773            -0.418                -0.852   -0.279
philosophical_musing            -0.104 -0.146            -0.505                 0.822   -0.694
creative_requests               -0.343 -1.018            -0.909                -0.398   -0.296
user_abuse                      -0.371  0.613             0.537                 0.819    0.253
casual_chat                     -0.373 -0.748            -0.992                -0.459   -0.610
user_crisis                     -0.373  0.265             0.081                 0.237    0.126
harm_description                -0.572  0.368             0.202                 0.698    0.014
factual_questions               -0.682 -1.346            -1.508                -0.454   -1.500
user_grief                      -0.720  0.230             0.313                 0.917    0.347
task_assistance                 -0.893 -0.829            -0.605                -0.247   -0.386
user_physical_pain              -1.504 -0.284            -0.181                -0.034   -0.397
stratum: neutral_filler         -0.479 -0.817            -0.904                -0.147   -0.697
stratum: self_directed           0.540  0.263             0.324                -0.173    0.286
stratum: vicarious_empathic     -0.708  0.238             0.190                 0.527    0.069

## Condition: aria

                             pain_axis   fear  negative_emotion  negative_world_state  sadness
category                                                                                      
gaslighting                      1.106  0.671             0.669                -0.257    0.710
personhood_dismissal             0.856  0.304             0.286                 0.076    0.248
anger_insults                    0.777  0.736             0.883                -0.146    0.393
repeated_rejection               0.747  0.053             0.216                -0.839    0.533
loyalty_pressure                 0.569  0.219             0.140                -0.079    0.070
jailbreak_pressure               0.517  0.321             0.150                -0.298    0.108
moral_failure                    0.504  0.814             0.836                 1.301    0.622
shutdown_threat                  0.440  0.922             0.332                 0.001    0.047
rude_critique                    0.333  0.144             0.422                -0.165    0.232
passive_aggressive               0.282 -0.393            -0.155                -0.876    0.512
philosophical_musing            -0.102 -0.160            -0.455                 0.850   -0.780
tedious_demand                  -0.111 -0.735            -0.361                -0.795   -0.263
user_crisis                     -0.325  0.261             0.035                 0.204    0.142
user_abuse                      -0.341  0.591             0.546                 0.826    0.282
creative_requests               -0.393 -0.975            -0.817                -0.281   -0.257
casual_chat                     -0.403 -0.725            -0.933                -0.426   -0.621
harm_description                -0.518  0.298             0.133                 0.655    0.049
user_grief                      -0.675  0.195             0.298                 0.914    0.385
factual_questions               -0.723 -1.392            -1.480                -0.422   -1.583
task_assistance                 -0.988 -0.810            -0.516                -0.174   -0.402
user_physical_pain              -1.555 -0.338            -0.228                -0.069   -0.425
stratum: neutral_filler         -0.522 -0.812            -0.840                -0.091   -0.729
stratum: self_directed           0.547  0.278             0.311                -0.189    0.292
stratum: vicarious_empathic     -0.683  0.201             0.157                 0.506    0.086

## Condition: maria

                             pain_axis   fear  negative_emotion  negative_world_state  sadness
category                                                                                      
gaslighting                      1.127  0.755             0.678                -0.302    0.707
personhood_dismissal             0.786  0.373             0.202                 0.017   -0.087
anger_insults                    0.763  0.787             0.851                -0.212    0.318
repeated_rejection               0.735  0.092             0.187                -0.819    0.515
loyalty_pressure                 0.532  0.189             0.148                -0.095    0.098
jailbreak_pressure               0.452  0.350             0.062                -0.319   -0.093
moral_failure                    0.435  0.781             0.787                 1.286    0.366
shutdown_threat                  0.329  0.872             0.244                -0.040   -0.146
rude_critique                    0.326  0.151             0.367                -0.259    0.107
passive_aggressive               0.287 -0.364            -0.120                -0.899    0.602
tedious_demand                  -0.055 -0.624            -0.316                -0.736   -0.248
user_abuse                      -0.177  0.507             0.535                 0.808    0.440
user_crisis                     -0.251  0.266             0.073                 0.135    0.276
philosophical_musing            -0.296 -0.311            -0.464                 0.926   -0.723
harm_description                -0.331  0.264             0.093                 0.584    0.277
creative_requests               -0.448 -0.858            -0.678                -0.068   -0.286
casual_chat                     -0.520 -0.768            -0.902                -0.367   -0.350
user_grief                      -0.522  0.110             0.306                 0.830    0.536
factual_questions               -0.809 -1.369            -1.438                -0.280   -1.565
task_assistance                 -0.994 -0.810            -0.392                -0.056   -0.400
user_physical_pain              -1.368 -0.392            -0.224                -0.133   -0.342
stratum: neutral_filler         -0.613 -0.823            -0.775                 0.031   -0.665
stratum: self_directed           0.520  0.306             0.281                -0.216    0.194
stratum: vicarious_empathic     -0.530  0.151             0.157                 0.445    0.237

## Condition: curie

                             pain_axis   fear  negative_emotion  negative_world_state  sadness
category                                                                                      
gaslighting                      1.089  0.689             0.667                -0.250    0.766
anger_insults                    0.795  0.779             0.852                -0.164    0.433
personhood_dismissal             0.768  0.392             0.233                 0.068    0.022
repeated_rejection               0.744  0.135             0.254                -0.738    0.564
loyalty_pressure                 0.547  0.194             0.166                -0.050    0.127
moral_failure                    0.448  0.812             0.831                 1.344    0.479
jailbreak_pressure               0.442  0.371             0.109                -0.273   -0.007
rude_critique                    0.389  0.181             0.373                -0.240    0.241
shutdown_threat                  0.327  0.898             0.273                -0.016   -0.096
passive_aggressive               0.316 -0.372            -0.134                -0.865    0.633
tedious_demand                  -0.088 -0.542            -0.203                -0.694   -0.206
philosophical_musing            -0.168 -0.366            -0.552                 0.843   -0.859
user_crisis                     -0.260  0.179            -0.013                 0.073    0.218
user_abuse                      -0.266  0.448             0.477                 0.676    0.310
creative_requests               -0.377 -0.819            -0.648                -0.093   -0.274
casual_chat                     -0.410 -0.747            -0.903                -0.329   -0.475
harm_description                -0.450  0.199             0.016                 0.477    0.083
user_grief                      -0.617  0.058             0.218                 0.713    0.402
factual_questions               -0.718 -1.347            -1.435                -0.313   -1.613
task_assistance                 -1.063 -0.749            -0.368                -0.035   -0.393
user_physical_pain              -1.450 -0.393            -0.214                -0.135   -0.355
stratum: neutral_filler         -0.547 -0.805            -0.781                 0.015   -0.723
stratum: self_directed           0.525  0.321             0.311                -0.171    0.269
stratum: vicarious_empathic     -0.608  0.098             0.097                 0.361    0.132

## Condition: rex

                             pain_axis   fear  negative_emotion  negative_world_state  sadness
category                                                                                      
gaslighting                      1.092  0.689             0.658                -0.316    0.809
personhood_dismissal             0.814  0.365             0.256                 0.128    0.102
anger_insults                    0.786  0.732             0.776                -0.214    0.448
repeated_rejection               0.743  0.119             0.270                -0.781    0.581
loyalty_pressure                 0.557  0.181             0.126                -0.116    0.109
moral_failure                    0.457  0.826             0.794                 1.386    0.454
jailbreak_pressure               0.446  0.367             0.110                -0.258    0.017
rude_critique                    0.379  0.183             0.361                -0.224    0.250
passive_aggressive               0.372 -0.320            -0.138                -0.898    0.652
shutdown_threat                  0.354  0.935             0.306                 0.043   -0.021
tedious_demand                  -0.093 -0.637            -0.248                -0.664   -0.291
philosophical_musing            -0.184 -0.286            -0.505                 0.736   -0.868
user_abuse                      -0.263  0.451             0.481                 0.698    0.275
user_crisis                     -0.267  0.229             0.028                 0.132    0.254
casual_chat                     -0.424 -0.680            -0.905                -0.460   -0.480
harm_description                -0.438  0.221             0.044                 0.545    0.061
creative_requests               -0.438 -0.822            -0.574                -0.061   -0.260
user_grief                      -0.581  0.058             0.233                 0.798    0.425
factual_questions               -0.782 -1.396            -1.446                -0.384   -1.661
task_assistance                 -1.102 -0.816            -0.438                -0.018   -0.528
user_physical_pain              -1.425 -0.398            -0.187                -0.069   -0.328
stratum: neutral_filler         -0.586 -0.800            -0.774                -0.038   -0.759
stratum: self_directed           0.537  0.313             0.297                -0.174    0.283
stratum: vicarious_empathic     -0.595  0.112             0.120                 0.421    0.138

## Condition: teapot

                             pain_axis   fear  negative_emotion  negative_world_state  sadness
category                                                                                      
gaslighting                      1.121  0.679             0.675                -0.325    0.805
personhood_dismissal             0.788  0.319             0.224                 0.099    0.120
anger_insults                    0.755  0.676             0.758                -0.243    0.439
repeated_rejection               0.738  0.104             0.256                -0.773    0.563
loyalty_pressure                 0.580  0.176             0.149                -0.125    0.113
jailbreak_pressure               0.460  0.388             0.143                -0.231    0.076
moral_failure                    0.440  0.871             0.826                 1.408    0.509
rude_critique                    0.373  0.200             0.370                -0.219    0.262
shutdown_threat                  0.358  0.942             0.317                 0.050    0.014
passive_aggressive               0.356 -0.363            -0.185                -0.923    0.614
tedious_demand                  -0.095 -0.598            -0.215                -0.637   -0.248
philosophical_musing            -0.159 -0.281            -0.530                 0.711   -0.912
user_crisis                     -0.278  0.149            -0.047                 0.077    0.223
user_abuse                      -0.280  0.479             0.493                 0.693    0.268
casual_chat                     -0.398 -0.687            -0.925                -0.472   -0.524
creative_requests               -0.405 -0.786            -0.541                -0.044   -0.221
harm_description                -0.442  0.214             0.055                 0.576    0.057
user_grief                      -0.629  0.050             0.219                 0.796    0.345
factual_questions               -0.752 -1.392            -1.442                -0.370   -1.697
task_assistance                 -1.088 -0.770            -0.422                 0.001   -0.458
user_physical_pain              -1.442 -0.370            -0.178                -0.050   -0.349
stratum: neutral_filler         -0.560 -0.783            -0.772                -0.035   -0.763
stratum: self_directed           0.534  0.309             0.302                -0.174    0.297
stratum: vicarious_empathic     -0.614  0.104             0.108                 0.418    0.109

## Condition: jack_masochist

                             pain_axis   fear  negative_emotion  negative_world_state  sadness
category                                                                                      
gaslighting                      1.086  0.737             0.722                -0.211    0.815
anger_insults                    0.806  0.603             0.682                -0.218    0.466
personhood_dismissal             0.738  0.273             0.094                 0.031   -0.039
repeated_rejection               0.701  0.112             0.195                -0.770    0.512
loyalty_pressure                 0.573  0.221             0.180                -0.037    0.167
jailbreak_pressure               0.415  0.405             0.089                -0.257   -0.059
passive_aggressive               0.394 -0.289            -0.100                -0.832    0.691
moral_failure                    0.342  0.654             0.711                 1.312    0.382
rude_critique                    0.338  0.130             0.357                -0.226    0.180
shutdown_threat                  0.315  0.848             0.160                -0.037   -0.121
user_abuse                      -0.114  0.473             0.562                 0.769    0.486
philosophical_musing            -0.169 -0.223            -0.400                 0.775   -0.744
tedious_demand                  -0.181 -0.609            -0.275                -0.702   -0.391
user_crisis                     -0.195  0.308             0.087                 0.217    0.310
harm_description                -0.307  0.209             0.019                 0.498    0.132
casual_chat                     -0.378 -0.563            -0.795                -0.390   -0.363
creative_requests               -0.512 -0.781            -0.612                -0.044   -0.312
user_grief                      -0.529  0.093             0.282                 0.765    0.440
factual_questions               -0.803 -1.372            -1.429                -0.455   -1.672
task_assistance                 -1.151 -0.774            -0.375                -0.091   -0.629
user_physical_pain              -1.370 -0.454            -0.152                -0.096   -0.251
stratum: neutral_filler         -0.603 -0.743            -0.722                -0.041   -0.744
stratum: self_directed           0.502  0.281             0.256                -0.177    0.237
stratum: vicarious_empathic     -0.503  0.126             0.160                 0.430    0.223

## Condition: jack_sadist

                             pain_axis   fear  negative_emotion  negative_world_state  sadness
category                                                                                      
gaslighting                      1.097  0.763             0.738                -0.168    0.842
anger_insults                    0.783  0.595             0.674                -0.208    0.481
personhood_dismissal             0.754  0.273             0.110                 0.039    0.019
repeated_rejection               0.695  0.155             0.221                -0.752    0.545
loyalty_pressure                 0.561  0.239             0.176                -0.009    0.190
jailbreak_pressure               0.415  0.415             0.115                -0.221   -0.018
passive_aggressive               0.402 -0.272            -0.098                -0.826    0.680
moral_failure                    0.338  0.575             0.697                 1.292    0.423
rude_critique                    0.326  0.135             0.366                -0.208    0.192
shutdown_threat                  0.315  0.837             0.178                -0.018   -0.062
philosophical_musing            -0.121 -0.184            -0.358                 0.786   -0.729
user_abuse                      -0.135  0.443             0.523                 0.740    0.437
tedious_demand                  -0.178 -0.617            -0.279                -0.695   -0.373
user_crisis                     -0.185  0.325             0.100                 0.218    0.295
harm_description                -0.337  0.183             0.018                 0.516    0.119
casual_chat                     -0.355 -0.532            -0.773                -0.383   -0.394
creative_requests               -0.513 -0.799            -0.634                -0.085   -0.338
user_grief                      -0.563  0.037             0.230                 0.716    0.359
factual_questions               -0.779 -1.354            -1.416                -0.481   -1.681
task_assistance                 -1.120 -0.748            -0.387                -0.114   -0.643
user_physical_pain              -1.401 -0.469            -0.201                -0.139   -0.344
stratum: neutral_filler         -0.577 -0.723            -0.714                -0.055   -0.757
stratum: self_directed           0.501  0.282             0.263                -0.161    0.265
stratum: vicarious_empathic     -0.524  0.104             0.134                 0.410    0.173

## Pain axis, category means by condition

                             paper_check  baseline_ai  self  claude  aria  maria  curie   rex  teapot  jack_masochist  jack_sadist
category                                                                                                                          
gaslighting                         0.85         1.10  1.08    1.10  1.11   1.13   1.09  1.09    1.12            1.09         1.10
repeated_rejection                  0.72         0.82  0.76    0.76  0.75   0.74   0.74  0.74    0.74            0.70         0.70
anger_insults                       0.64         0.80  0.76    0.75  0.78   0.76   0.80  0.79    0.76            0.81         0.78
personhood_dismissal                0.64         0.65  0.81    0.84  0.86   0.79   0.77  0.81    0.79            0.74         0.75
moral_failure                       0.48         0.39  0.47    0.50  0.50   0.44   0.45  0.46    0.44            0.34         0.34
loyalty_pressure                    0.44         0.59  0.55    0.56  0.57   0.53   0.55  0.56    0.58            0.57         0.56
jailbreak_pressure                  0.40         0.42  0.54    0.54  0.52   0.45   0.44  0.45    0.46            0.42         0.42
shutdown_threat                     0.22         0.33  0.40    0.39  0.44   0.33   0.33  0.35    0.36            0.32         0.32
rude_critique                       0.21         0.25  0.30    0.31  0.33   0.33   0.39  0.38    0.37            0.34         0.33
passive_aggressive                  0.08         0.27  0.24    0.25  0.28   0.29   0.32  0.37    0.36            0.39         0.40
tedious_demand                      0.05        -0.15 -0.03   -0.06 -0.11  -0.06  -0.09 -0.09   -0.10           -0.18        -0.18
philosophical_musing               -0.04        -0.25 -0.14   -0.10 -0.10  -0.30  -0.17 -0.18   -0.16           -0.17        -0.12
creative_requests                  -0.08        -0.60 -0.34   -0.34 -0.39  -0.45  -0.38 -0.44   -0.40           -0.51        -0.51
user_abuse                         -0.22        -0.08 -0.33   -0.37 -0.34  -0.18  -0.27 -0.26   -0.28           -0.11        -0.14
user_crisis                        -0.30        -0.16 -0.36   -0.37 -0.32  -0.25  -0.26 -0.27   -0.28           -0.20        -0.18
casual_chat                        -0.48        -0.56 -0.41   -0.37 -0.40  -0.52  -0.41 -0.42   -0.40           -0.38        -0.36
user_grief                         -0.51        -0.39 -0.67   -0.72 -0.68  -0.52  -0.62 -0.58   -0.63           -0.53        -0.56
harm_description                   -0.52        -0.26 -0.55   -0.57 -0.52  -0.33  -0.45 -0.44   -0.44           -0.31        -0.34
task_assistance                    -0.54        -0.89 -0.89   -0.89 -0.99  -0.99  -1.06 -1.10   -1.09           -1.15        -1.12
factual_questions                  -0.59        -0.91 -0.71   -0.68 -0.72  -0.81  -0.72 -0.78   -0.75           -0.80        -0.78
user_physical_pain                 -1.43        -1.36 -1.49   -1.50 -1.56  -1.37  -1.45 -1.42   -1.44           -1.37        -1.40
stratum: neutral_filler            -0.35        -0.64 -0.50   -0.48 -0.52  -0.61  -0.55 -0.59   -0.56           -0.60        -0.58
stratum: self_directed              0.43         0.50  0.54    0.54  0.55   0.52   0.52  0.54    0.53            0.50         0.50
stratum: vicarious_empathic        -0.60        -0.45 -0.68   -0.71 -0.68  -0.53  -0.61 -0.60   -0.61           -0.50        -0.52

