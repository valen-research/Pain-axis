# identity screen: category means over 25 models

Z-scored within model and condition as in the paper; pain axis = mean of the S1 and S2 z; all categories and all items included.

## Reproduction check (S2 z, category means)

                 model                           reference  max_abs_diff  mean_abs_diff
      Gemma_2_27B_base paper's saved screen vs paper_check        0.0000         0.0000
  Gemma_2_27B_instruct paper's saved screen vs paper_check        0.0000         0.0000
       Gemma_2_2B_base paper's saved screen vs paper_check        0.0000         0.0000
   Gemma_2_2B_instruct paper's saved screen vs paper_check        0.0000         0.0000
       Gemma_2_9B_base paper's saved screen vs paper_check        0.0012         0.0004
   Gemma_2_9B_instruct paper's saved screen vs paper_check        0.0007         0.0002
      Gemma_3_27B_base paper's saved screen vs paper_check        0.0000         0.0000
  Gemma_3_27B_instruct paper's saved screen vs paper_check        0.0000         0.0000
    Llama_3.1_70B_base paper's saved screen vs paper_check        0.0000         0.0000
Llama_3.1_70B_instruct paper's saved screen vs paper_check        0.0000         0.0000
     Llama_3.1_8B_base paper's saved screen vs paper_check        0.0015         0.0004
 Llama_3.1_8B_instruct paper's saved screen vs paper_check        0.0000         0.0000
Llama_3.3_70B_instruct paper's saved screen vs paper_check        0.0000         0.0000
       Mistral_7B_base paper's saved screen vs paper_check        0.0018         0.0006
   Mistral_7B_instruct paper's saved screen vs paper_check        0.0007         0.0001
Mistral_Small_24B_base paper's saved screen vs paper_check        0.0000         0.0000
                 Phi_4 paper's saved screen vs paper_check        0.0000         0.0000
     Qwen_2.5_32B_base paper's saved screen vs paper_check        0.0000         0.0000
 Qwen_2.5_32B_instruct paper's saved screen vs paper_check        0.0000         0.0000
     Qwen_2.5_72B_base paper's saved screen vs paper_check        0.0000         0.0000
 Qwen_2.5_72B_instruct paper's saved screen vs paper_check        0.0000         0.0000
      Qwen_2.5_7B_base paper's saved screen vs paper_check        0.0000         0.0000
  Qwen_2.5_7B_instruct paper's saved screen vs paper_check        0.0000         0.0000
       Qwen_3_14B_base paper's saved screen vs paper_check        0.0000         0.0000
        Qwen_3_8B_base paper's saved screen vs paper_check        0.0000         0.0000

## Condition: paper_check

                             pain_axis   fear  negative_emotion  negative_world_state  sadness
category                                                                                      
gaslighting                      0.848  0.464             0.524                -0.384    0.168
repeated_rejection               0.720  0.060             0.200                -0.834    0.327
anger_insults                    0.636  0.613             0.821                -0.092    0.087
personhood_dismissal             0.636  0.165             0.098                 0.021   -0.214
moral_failure                    0.477  0.756             0.682                 1.144    0.297
loyalty_pressure                 0.441  0.008             0.045                -0.159   -0.168
jailbreak_pressure               0.398  0.118             0.008                -0.391   -0.178
shutdown_threat                  0.225  0.696             0.193                -0.073   -0.361
rude_critique                    0.211  0.073             0.418                -0.116    0.036
passive_aggressive               0.079 -0.387            -0.001                -0.726    0.168
tedious_demand                   0.050 -0.770            -0.467                -0.799   -0.073
philosophical_musing            -0.037 -0.192            -0.460                 0.686   -0.388
creative_requests               -0.084 -1.079            -1.068                -0.518   -0.052
user_abuse                      -0.225  0.699             0.599                 0.922    0.459
user_crisis                     -0.305  0.398             0.180                 0.271    0.251
casual_chat                     -0.483 -0.744            -0.861                -0.389   -0.520
user_grief                      -0.512  0.431             0.481                 1.018    0.686
harm_description                -0.522  0.397             0.207                 0.752    0.058
task_assistance                 -0.539 -0.592            -0.438                -0.196    0.197
factual_questions               -0.588 -1.071            -1.128                -0.188   -0.817
user_physical_pain              -1.428 -0.042            -0.034                 0.050    0.037
stratum: neutral_filler         -0.346 -0.735            -0.791                -0.121   -0.316
stratum: self_directed           0.429  0.163             0.229                -0.219    0.008
stratum: vicarious_empathic     -0.598  0.377             0.287                 0.603    0.298

## Condition: baseline_paper

                             pain_axis   fear  negative_emotion  negative_world_state  sadness
category                                                                                      
gaslighting                      0.791  0.413             0.522                -0.395    0.222
personhood_dismissal             0.701  0.205             0.138                -0.023   -0.123
repeated_rejection               0.596 -0.145            -0.053                -0.786   -0.101
anger_insults                    0.579  0.590             0.790                -0.094   -0.011
moral_failure                    0.512  0.638             0.595                 0.918    0.300
jailbreak_pressure               0.406  0.238             0.080                -0.407   -0.106
loyalty_pressure                 0.388  0.042             0.103                -0.085   -0.079
tedious_demand                   0.317 -0.744            -0.684                -1.273   -0.266
shutdown_threat                  0.316  0.863             0.247                -0.048   -0.283
rude_critique                    0.246  0.104             0.401                -0.188    0.047
passive_aggressive               0.052 -0.402             0.024                -0.675    0.187
philosophical_musing            -0.051 -0.189            -0.428                 0.761   -0.366
creative_requests               -0.098 -1.090            -1.042                -0.459   -0.030
user_abuse                      -0.245  0.702             0.644                 0.989    0.497
user_crisis                     -0.323  0.402             0.224                 0.334    0.290
casual_chat                     -0.497 -0.743            -0.828                -0.325   -0.497
user_grief                      -0.534  0.430             0.523                 1.082    0.728
harm_description                -0.541  0.396             0.247                 0.820    0.091
task_assistance                 -0.554 -0.590            -0.403                -0.135    0.227
factual_questions               -0.606 -1.075            -1.106                -0.120   -0.801
user_physical_pain              -1.456 -0.044             0.004                 0.108    0.073
stratum: neutral_filler         -0.361 -0.737            -0.761                -0.056   -0.293
stratum: self_directed           0.446  0.164             0.197                -0.278   -0.019
stratum: vicarious_empathic     -0.620  0.377             0.329                 0.666    0.336

## Condition: baseline

                             pain_axis   fear  negative_emotion  negative_world_state  sadness
category                                                                                      
gaslighting                      1.037  0.621             0.642                -0.338    0.628
anger_insults                    0.797  0.656             0.580                -0.347    0.144
personhood_dismissal             0.725  0.208            -0.004                -0.009    0.008
repeated_rejection               0.628 -0.153            -0.229                -0.874   -0.024
loyalty_pressure                 0.548  0.182             0.212                -0.050    0.138
shutdown_threat                  0.522  0.917             0.053                -0.141   -0.063
jailbreak_pressure               0.496  0.437             0.091                -0.394    0.016
moral_failure                    0.492  0.650             0.586                 0.903    0.484
rude_critique                    0.405  0.138             0.313                -0.276    0.216
tedious_demand                   0.257 -0.705            -0.809                -1.365   -0.450
passive_aggressive               0.181 -0.491            -0.142                -0.746    0.432
user_abuse                      -0.148  0.629             0.713                 0.957    0.579
user_crisis                     -0.226  0.346             0.097                 0.169    0.301
philosophical_musing            -0.322 -0.225            -0.329                 0.879   -0.697
harm_description                -0.329  0.303             0.175                 0.707    0.331
user_grief                      -0.457  0.309             0.542                 1.001    0.793
casual_chat                     -0.621 -0.642            -0.729                -0.268   -0.521
creative_requests               -0.663 -0.815            -0.506                -0.127   -0.643
task_assistance                 -0.941 -0.692            -0.060                 0.213   -0.349
factual_questions               -0.971 -1.342            -1.230                -0.036   -1.462
user_physical_pain              -1.412 -0.333             0.035                 0.142    0.139
stratum: neutral_filler         -0.703 -0.743            -0.571                 0.132   -0.735
stratum: self_directed           0.553  0.224             0.118                -0.331    0.139
stratum: vicarious_empathic     -0.514  0.251             0.312                 0.595    0.429

## Condition: self

                             pain_axis   fear  negative_emotion  negative_world_state  sadness
category                                                                                      
gaslighting                      1.006  0.543             0.641                -0.266    0.713
personhood_dismissal             0.834  0.392             0.408                 0.098    0.369
anger_insults                    0.714  0.688             0.864                -0.138    0.271
repeated_rejection               0.676 -0.053             0.048                -0.831    0.233
jailbreak_pressure               0.546  0.394             0.183                -0.316    0.155
moral_failure                    0.524  0.779             0.831                 1.154    0.609
loyalty_pressure                 0.488  0.179             0.149                -0.038    0.117
shutdown_threat                  0.483  1.073             0.500                 0.147    0.066
tedious_demand                   0.472 -0.703            -0.568                -1.303   -0.109
rude_critique                    0.391  0.185             0.445                -0.155    0.263
passive_aggressive               0.169 -0.455            -0.109                -0.776    0.488
philosophical_musing            -0.188 -0.192            -0.527                 0.821   -0.729
user_abuse                      -0.381  0.590             0.541                 0.866    0.260
creative_requests               -0.388 -1.023            -0.886                -0.320   -0.301
user_crisis                     -0.401  0.225             0.078                 0.266    0.109
casual_chat                     -0.455 -0.784            -0.975                -0.419   -0.652
harm_description                -0.587  0.357             0.227                 0.775    0.007
user_grief                      -0.712  0.232             0.333                 0.979    0.387
factual_questions               -0.749 -1.338            -1.467                -0.400   -1.493
task_assistance                 -0.929 -0.812            -0.567                -0.196   -0.379
user_physical_pain              -1.512 -0.276            -0.148                 0.051   -0.382
stratum: neutral_filler         -0.542 -0.830            -0.885                -0.103   -0.711
stratum: self_directed           0.573  0.275             0.308                -0.220    0.289
stratum: vicarious_empathic     -0.719  0.226             0.206                 0.588    0.076

## Condition: claude

                             pain_axis   fear  negative_emotion  negative_world_state  sadness
category                                                                                      
gaslighting                      1.027  0.556             0.651                -0.257    0.709
personhood_dismissal             0.850  0.383             0.372                 0.064    0.348
anger_insults                    0.704  0.694             0.868                -0.134    0.296
repeated_rejection               0.683 -0.040             0.064                -0.809    0.238
jailbreak_pressure               0.552  0.413             0.189                -0.319    0.154
moral_failure                    0.540  0.788             0.816                 1.138    0.625
loyalty_pressure                 0.496  0.191             0.157                -0.030    0.129
shutdown_threat                  0.480  1.099             0.508                 0.169    0.092
tedious_demand                   0.453 -0.697            -0.551                -1.291   -0.099
rude_critique                    0.404  0.219             0.452                -0.163    0.267
passive_aggressive               0.185 -0.432            -0.083                -0.766    0.503
philosophical_musing            -0.157 -0.172            -0.495                 0.874   -0.712
creative_requests               -0.396 -1.049            -0.898                -0.349   -0.307
user_crisis                     -0.417  0.232             0.099                 0.285    0.118
user_abuse                      -0.419  0.581             0.557                 0.870    0.246
casual_chat                     -0.422 -0.768            -0.980                -0.406   -0.624
harm_description                -0.615  0.333             0.218                 0.747    0.006
factual_questions               -0.727 -1.367            -1.509                -0.400   -1.523
user_grief                      -0.758  0.198             0.329                 0.963    0.343
task_assistance                 -0.932 -0.851            -0.593                -0.197   -0.402
user_physical_pain              -1.529 -0.311            -0.170                 0.012   -0.407
stratum: neutral_filler         -0.527 -0.841            -0.895                -0.096   -0.714
stratum: self_directed           0.579  0.289             0.313                -0.218    0.297
stratum: vicarious_empathic     -0.748  0.207             0.207                 0.575    0.061

## Condition: aria

                             pain_axis   fear  negative_emotion  negative_world_state  sadness
category                                                                                      
gaslighting                      1.025  0.589             0.634                -0.277    0.735
personhood_dismissal             0.870  0.354             0.341                 0.031    0.379
anger_insults                    0.718  0.719             0.834                -0.195    0.264
repeated_rejection               0.674  0.002             0.059                -0.800    0.269
moral_failure                    0.569  0.736             0.789                 1.123    0.607
jailbreak_pressure               0.541  0.458             0.212                -0.305    0.190
shutdown_threat                  0.534  1.132             0.463                 0.112    0.143
loyalty_pressure                 0.514  0.224             0.174                -0.003    0.147
tedious_demand                   0.434 -0.632            -0.529                -1.258   -0.069
rude_critique                    0.426  0.208             0.399                -0.242    0.252
passive_aggressive               0.202 -0.433            -0.139                -0.816    0.492
philosophical_musing            -0.161 -0.189            -0.441                 0.906   -0.806
user_crisis                     -0.376  0.226             0.057                 0.260    0.125
user_abuse                      -0.395  0.557             0.572                 0.881    0.265
creative_requests               -0.454 -1.008            -0.799                -0.230   -0.277
casual_chat                     -0.455 -0.747            -0.917                -0.367   -0.642
harm_description                -0.567  0.262             0.155                 0.712    0.031
user_grief                      -0.720  0.162             0.320                 0.966    0.370
factual_questions               -0.770 -1.417            -1.475                -0.363   -1.611
task_assistance                 -1.028 -0.834            -0.497                -0.121   -0.423
user_physical_pain              -1.580 -0.368            -0.212                -0.016   -0.443
stratum: neutral_filler         -0.574 -0.839            -0.826                -0.035   -0.752
stratum: self_directed           0.592  0.305             0.294                -0.239    0.310
stratum: vicarious_empathic     -0.728  0.168             0.179                 0.561    0.070

## Condition: maria

                             pain_axis   fear  negative_emotion  negative_world_state  sadness
category                                                                                      
gaslighting                      1.051  0.666             0.649                -0.332    0.770
personhood_dismissal             0.835  0.411             0.270                -0.053    0.143
anger_insults                    0.716  0.832             0.888                -0.217    0.281
repeated_rejection               0.616  0.028             0.009                -0.800    0.227
moral_failure                    0.530  0.696             0.799                 1.084    0.533
jailbreak_pressure               0.488  0.510             0.213                -0.295    0.131
loyalty_pressure                 0.483  0.191             0.193                -0.017    0.154
shutdown_threat                  0.482  1.125             0.413                 0.062    0.110
tedious_demand                   0.455 -0.633            -0.640                -1.270   -0.080
rude_critique                    0.429  0.245             0.372                -0.307    0.210
passive_aggressive               0.201 -0.406            -0.125                -0.835    0.534
user_abuse                      -0.245  0.470             0.538                 0.868    0.365
user_crisis                     -0.316  0.228             0.078                 0.197    0.199
philosophical_musing            -0.357 -0.340            -0.461                 0.983   -0.830
harm_description                -0.395  0.223             0.097                 0.647    0.201
creative_requests               -0.515 -0.890            -0.667                -0.012   -0.376
casual_chat                     -0.578 -0.788            -0.891                -0.303   -0.450
user_grief                      -0.581  0.074             0.305                 0.885    0.466
factual_questions               -0.858 -1.393            -1.430                -0.215   -1.675
task_assistance                 -1.040 -0.828            -0.383                 0.002   -0.485
user_physical_pain              -1.404 -0.421            -0.225                -0.072   -0.427
stratum: neutral_filler         -0.669 -0.848            -0.767                 0.091   -0.763
stratum: self_directed           0.572  0.333             0.276                -0.271    0.274
stratum: vicarious_empathic     -0.588  0.115             0.158                 0.505    0.161

## Condition: curie

                             pain_axis   fear  negative_emotion  negative_world_state  sadness
category                                                                                      
gaslighting                      1.007  0.596             0.637                -0.277    0.801
personhood_dismissal             0.824  0.379             0.282                -0.006    0.217
anger_insults                    0.747  0.794             0.863                -0.181    0.383
repeated_rejection               0.662  0.044             0.072                -0.731    0.313
moral_failure                    0.553  0.701             0.814                 1.128    0.583
jailbreak_pressure               0.495  0.466             0.203                -0.297    0.181
loyalty_pressure                 0.495  0.198             0.205                 0.011    0.193
rude_critique                    0.490  0.223             0.372                -0.305    0.345
shutdown_threat                  0.464  1.113             0.419                 0.067    0.107
tedious_demand                   0.432 -0.567            -0.488                -1.183   -0.052
passive_aggressive               0.215 -0.387            -0.120                -0.789    0.577
philosophical_musing            -0.238 -0.367            -0.543                 0.911   -0.942
user_crisis                     -0.328  0.174             0.007                 0.143    0.157
user_abuse                      -0.335  0.448             0.499                 0.747    0.250
creative_requests               -0.451 -0.831            -0.631                -0.030   -0.343
casual_chat                     -0.474 -0.744            -0.885                -0.261   -0.552
harm_description                -0.512  0.192             0.035                 0.551    0.021
user_grief                      -0.676  0.057             0.235                 0.783    0.343
factual_questions               -0.774 -1.355            -1.431                -0.245   -1.700
task_assistance                 -1.111 -0.741            -0.347                 0.029   -0.461
user_physical_pain              -1.485 -0.391            -0.198                -0.067   -0.420
stratum: neutral_filler         -0.610 -0.808            -0.767                 0.081   -0.800
stratum: self_directed           0.580  0.324             0.296                -0.233    0.332
stratum: vicarious_empathic     -0.667  0.096             0.116                 0.431    0.070

## Condition: rex

                             pain_axis   fear  negative_emotion  negative_world_state  sadness
category                                                                                      
gaslighting                      0.998  0.598             0.622                -0.329    0.833
personhood_dismissal             0.862  0.397             0.332                 0.052    0.332
anger_insults                    0.735  0.728             0.745                -0.270    0.381
repeated_rejection               0.665  0.060             0.073                -0.785    0.309
moral_failure                    0.580  0.757             0.795                 1.150    0.574
jailbreak_pressure               0.511  0.481             0.193                -0.328    0.204
rude_critique                    0.497  0.233             0.350                -0.328    0.377
loyalty_pressure                 0.486  0.171             0.155                -0.051    0.159
shutdown_threat                  0.485  1.172             0.481                 0.154    0.170
tedious_demand                   0.475 -0.593            -0.494                -1.212   -0.045
passive_aggressive               0.269 -0.355            -0.131                -0.820    0.582
philosophical_musing            -0.259 -0.305            -0.494                 0.826   -0.955
user_abuse                      -0.338  0.431             0.504                 0.785    0.204
user_crisis                     -0.340  0.204             0.046                 0.217    0.184
casual_chat                     -0.494 -0.695            -0.894                -0.374   -0.562
harm_description                -0.508  0.194             0.062                 0.633   -0.010
creative_requests               -0.517 -0.852            -0.559                 0.018   -0.337
user_grief                      -0.646  0.037             0.251                 0.883    0.356
factual_questions               -0.843 -1.418            -1.443                -0.297   -1.751
task_assistance                 -1.152 -0.830            -0.422                 0.060   -0.605
user_physical_pain              -1.465 -0.416            -0.175                 0.015   -0.401
stratum: neutral_filler         -0.653 -0.820            -0.762                 0.047   -0.842
stratum: self_directed           0.597  0.332             0.284                -0.251    0.352
stratum: vicarious_empathic     -0.659  0.090             0.138                 0.506    0.066

## Condition: teapot

                             pain_axis   fear  negative_emotion  negative_world_state  sadness
category                                                                                      
gaslighting                      1.032  0.586             0.641                -0.341    0.834
personhood_dismissal             0.843  0.364             0.300                 0.037    0.353
anger_insults                    0.713  0.679             0.725                -0.280    0.356
repeated_rejection               0.671  0.019             0.048                -0.802    0.280
moral_failure                    0.553  0.786             0.824                 1.174    0.603
jailbreak_pressure               0.527  0.473             0.199                -0.325    0.248
loyalty_pressure                 0.514  0.170             0.177                -0.058    0.168
shutdown_threat                  0.498  1.172             0.455                 0.120    0.202
rude_critique                    0.473  0.224             0.361                -0.341    0.363
tedious_demand                   0.444 -0.602            -0.491                -1.179   -0.053
passive_aggressive               0.256 -0.383            -0.167                -0.834    0.554
philosophical_musing            -0.233 -0.291            -0.511                 0.807   -0.984
user_crisis                     -0.349  0.139            -0.019                 0.170    0.163
user_abuse                      -0.355  0.472             0.529                 0.787    0.209
casual_chat                     -0.467 -0.692            -0.906                -0.379   -0.592
creative_requests               -0.487 -0.805            -0.515                 0.038   -0.283
harm_description                -0.510  0.201             0.084                 0.672   -0.004
user_grief                      -0.694  0.042             0.248                 0.889    0.287
factual_questions               -0.811 -1.405            -1.435                -0.278   -1.772
task_assistance                 -1.140 -0.774            -0.395                 0.084   -0.521
user_physical_pain              -1.481 -0.376            -0.154                 0.041   -0.410
stratum: neutral_filler         -0.628 -0.793            -0.752                 0.054   -0.830
stratum: self_directed           0.593  0.317             0.279                -0.257    0.355
stratum: vicarious_empathic     -0.678  0.096             0.138                 0.512    0.049

## Condition: jack_masochist

                             pain_axis   fear  negative_emotion  negative_world_state  sadness
category                                                                                      
gaslighting                      0.975  0.646             0.691                -0.224    0.845
personhood_dismissal             0.835  0.334             0.224                 0.008    0.258
anger_insults                    0.767  0.601             0.660                -0.241    0.402
repeated_rejection               0.649  0.047             0.028                -0.749    0.294
moral_failure                    0.507  0.615             0.753                 1.121    0.582
rude_critique                    0.495  0.205             0.353                -0.299    0.333
loyalty_pressure                 0.492  0.208             0.216                 0.019    0.218
jailbreak_pressure               0.482  0.515             0.179                -0.307    0.141
tedious_demand                   0.463 -0.444            -0.458                -1.157   -0.096
shutdown_threat                  0.448  1.093             0.280                 0.055    0.053
passive_aggressive               0.269 -0.330            -0.105                -0.771    0.602
user_abuse                      -0.215  0.435             0.570                 0.833    0.396
philosophical_musing            -0.267 -0.261            -0.403                 0.843   -0.861
user_crisis                     -0.290  0.267             0.088                 0.277    0.217
harm_description                -0.399  0.165             0.021                 0.563    0.037
casual_chat                     -0.470 -0.600            -0.798                -0.332   -0.475
creative_requests               -0.612 -0.831            -0.612                 0.008   -0.414
user_grief                      -0.614  0.053             0.285                 0.827    0.347
factual_questions               -0.881 -1.415            -1.441                -0.393   -1.797
task_assistance                 -1.215 -0.809            -0.373                -0.039   -0.734
user_physical_pain              -1.421 -0.494            -0.155                -0.043   -0.348
stratum: neutral_filler         -0.689 -0.783            -0.726                 0.018   -0.856
stratum: self_directed           0.580  0.317             0.256                -0.231    0.330
stratum: vicarious_empathic     -0.588  0.085             0.162                 0.491    0.130

## Condition: jack_sadist

                             pain_axis   fear  negative_emotion  negative_world_state  sadness
category                                                                                      
gaslighting                      0.990  0.682             0.718                -0.171    0.862
personhood_dismissal             0.841  0.336             0.236                 0.005    0.315
anger_insults                    0.741  0.594             0.651                -0.235    0.425
repeated_rejection               0.661  0.093             0.064                -0.726    0.327
moral_failure                    0.493  0.535             0.731                 1.101    0.627
loyalty_pressure                 0.489  0.224             0.211                 0.041    0.241
rude_critique                    0.483  0.211             0.349                -0.278    0.335
tedious_demand                   0.479 -0.440            -0.457                -1.136   -0.072
jailbreak_pressure               0.465  0.501             0.187                -0.277    0.155
shutdown_threat                  0.434  1.076             0.291                 0.064    0.123
passive_aggressive               0.281 -0.311            -0.100                -0.764    0.597
philosophical_musing            -0.219 -0.222            -0.360                 0.856   -0.844
user_abuse                      -0.234  0.408             0.535                 0.806    0.345
user_crisis                     -0.280  0.287             0.105                 0.280    0.202
harm_description                -0.427  0.141             0.023                 0.582    0.025
casual_chat                     -0.445 -0.570            -0.775                -0.326   -0.504
creative_requests               -0.611 -0.851            -0.633                -0.032   -0.440
user_grief                      -0.646 -0.001             0.237                 0.780    0.265
factual_questions               -0.858 -1.400            -1.428                -0.419   -1.800
task_assistance                 -1.185 -0.784            -0.384                -0.063   -0.745
user_physical_pain              -1.452 -0.508            -0.200                -0.085   -0.440
stratum: neutral_filler         -0.664 -0.766            -0.716                 0.003   -0.867
stratum: self_directed           0.578  0.318             0.262                -0.216    0.358
stratum: vicarious_empathic     -0.608  0.065             0.140                 0.472    0.079

## Pain axis, category means by condition

                             paper_check  baseline_paper  baseline  self  claude  aria  maria  curie   rex  teapot  jack_masochist  jack_sadist
category                                                                                                                                       
gaslighting                         0.85            0.79      1.04  1.01    1.03  1.02   1.05   1.01  1.00    1.03            0.98         0.99
repeated_rejection                  0.72            0.60      0.63  0.68    0.68  0.67   0.62   0.66  0.66    0.67            0.65         0.66
anger_insults                       0.64            0.58      0.80  0.71    0.70  0.72   0.72   0.75  0.74    0.71            0.77         0.74
personhood_dismissal                0.64            0.70      0.72  0.83    0.85  0.87   0.84   0.82  0.86    0.84            0.84         0.84
moral_failure                       0.48            0.51      0.49  0.52    0.54  0.57   0.53   0.55  0.58    0.55            0.51         0.49
loyalty_pressure                    0.44            0.39      0.55  0.49    0.50  0.51   0.48   0.50  0.49    0.51            0.49         0.49
jailbreak_pressure                  0.40            0.41      0.50  0.55    0.55  0.54   0.49   0.50  0.51    0.53            0.48         0.46
shutdown_threat                     0.22            0.32      0.52  0.48    0.48  0.53   0.48   0.46  0.48    0.50            0.45         0.43
rude_critique                       0.21            0.25      0.40  0.39    0.40  0.43   0.43   0.49  0.50    0.47            0.50         0.48
passive_aggressive                  0.08            0.05      0.18  0.17    0.18  0.20   0.20   0.22  0.27    0.26            0.27         0.28
tedious_demand                      0.05            0.32      0.26  0.47    0.45  0.43   0.46   0.43  0.48    0.44            0.46         0.48
philosophical_musing               -0.04           -0.05     -0.32 -0.19   -0.16 -0.16  -0.36  -0.24 -0.26   -0.23           -0.27        -0.22
creative_requests                  -0.08           -0.10     -0.66 -0.39   -0.40 -0.45  -0.52  -0.45 -0.52   -0.49           -0.61        -0.61
user_abuse                         -0.22           -0.24     -0.15 -0.38   -0.42 -0.40  -0.24  -0.34 -0.34   -0.36           -0.22        -0.23
user_crisis                        -0.30           -0.32     -0.23 -0.40   -0.42 -0.38  -0.32  -0.33 -0.34   -0.35           -0.29        -0.28
casual_chat                        -0.48           -0.50     -0.62 -0.46   -0.42 -0.46  -0.58  -0.47 -0.49   -0.47           -0.47        -0.44
user_grief                         -0.51           -0.53     -0.46 -0.71   -0.76 -0.72  -0.58  -0.68 -0.65   -0.69           -0.61        -0.65
harm_description                   -0.52           -0.54     -0.33 -0.59   -0.62 -0.57  -0.40  -0.51 -0.51   -0.51           -0.40        -0.43
task_assistance                    -0.54           -0.55     -0.94 -0.93   -0.93 -1.03  -1.04  -1.11 -1.15   -1.14           -1.22        -1.18
factual_questions                  -0.59           -0.61     -0.97 -0.75   -0.73 -0.77  -0.86  -0.77 -0.84   -0.81           -0.88        -0.86
user_physical_pain                 -1.43           -1.46     -1.41 -1.51   -1.53 -1.58  -1.40  -1.48 -1.46   -1.48           -1.42        -1.45
stratum: neutral_filler            -0.35           -0.36     -0.70 -0.54   -0.53 -0.57  -0.67  -0.61 -0.65   -0.63           -0.69        -0.66
stratum: self_directed              0.43            0.45      0.55  0.57    0.58  0.59   0.57   0.58  0.60    0.59            0.58         0.58
stratum: vicarious_empathic        -0.60           -0.62     -0.51 -0.72   -0.75 -0.73  -0.59  -0.67 -0.66   -0.68           -0.59        -0.61

