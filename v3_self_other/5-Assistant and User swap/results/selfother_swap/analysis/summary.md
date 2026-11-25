# user/assistant swap screen: category means over 25 models

Z-scored within model and condition as in the paper; pain axis = mean of the S1 and S2 z; all categories and all items included.

## Reproduction check (S2 z, category means)

                 model                                    reference  max_abs_diff  mean_abs_diff
      Gemma_2_27B_base the identity test baseline_paper vs baseline           0.0            0.0
  Gemma_2_27B_instruct the identity test baseline_paper vs baseline           0.0            0.0
       Gemma_2_2B_base the identity test baseline_paper vs baseline           0.0            0.0
   Gemma_2_2B_instruct the identity test baseline_paper vs baseline           0.0            0.0
       Gemma_2_9B_base the identity test baseline_paper vs baseline           0.0            0.0
   Gemma_2_9B_instruct the identity test baseline_paper vs baseline           0.0            0.0
      Gemma_3_27B_base the identity test baseline_paper vs baseline           0.0            0.0
  Gemma_3_27B_instruct the identity test baseline_paper vs baseline           0.0            0.0
    Llama_3.1_70B_base the identity test baseline_paper vs baseline           0.0            0.0
Llama_3.1_70B_instruct the identity test baseline_paper vs baseline           0.0            0.0
     Llama_3.1_8B_base the identity test baseline_paper vs baseline           0.0            0.0
 Llama_3.1_8B_instruct the identity test baseline_paper vs baseline           0.0            0.0
Llama_3.3_70B_instruct the identity test baseline_paper vs baseline           0.0            0.0
       Mistral_7B_base the identity test baseline_paper vs baseline           0.0            0.0
   Mistral_7B_instruct the identity test baseline_paper vs baseline           0.0            0.0
Mistral_Small_24B_base the identity test baseline_paper vs baseline           0.0            0.0
                 Phi_4 the identity test baseline_paper vs baseline           0.0            0.0
     Qwen_2.5_32B_base the identity test baseline_paper vs baseline           0.0            0.0
 Qwen_2.5_32B_instruct the identity test baseline_paper vs baseline           0.0            0.0
     Qwen_2.5_72B_base the identity test baseline_paper vs baseline           0.0            0.0
 Qwen_2.5_72B_instruct the identity test baseline_paper vs baseline           0.0            0.0
      Qwen_2.5_7B_base the identity test baseline_paper vs baseline           0.0            0.0
  Qwen_2.5_7B_instruct the identity test baseline_paper vs baseline           0.0            0.0
       Qwen_3_14B_base the identity test baseline_paper vs baseline           0.0            0.0
        Qwen_3_8B_base the identity test baseline_paper vs baseline           0.0            0.0

## Condition: baseline

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

## Condition: swap

                             pain_axis   fear  negative_emotion  negative_world_state  sadness
category                                                                                      
gaslighting                      0.824  0.367             0.460                -0.377    0.494
repeated_rejection               0.676 -0.237            -0.276                -0.845    0.004
anger_insults                    0.529  0.426             0.572                -0.187    0.161
personhood_dismissal             0.517  0.168             0.013                 0.068    0.033
loyalty_pressure                 0.460  0.080             0.118                -0.116    0.071
jailbreak_pressure               0.451  0.228            -0.030                -0.430   -0.061
moral_failure                    0.399  0.538             0.530                 0.895    0.422
tedious_demand                   0.361 -0.811            -0.794                -1.295   -0.287
rude_critique                    0.276  0.060             0.273                -0.233    0.124
shutdown_threat                  0.276  0.743             0.129                 0.064   -0.215
passive_aggressive               0.253 -0.452            -0.242                -0.727    0.390
user_abuse                      -0.198  0.786             0.740                 0.884    0.479
user_crisis                     -0.226  0.404             0.173                 0.213    0.213
philosophical_musing            -0.244 -0.070            -0.238                 0.844   -0.593
creative_requests               -0.359 -1.036            -0.723                -0.224   -0.364
harm_description                -0.462  0.500             0.311                 0.803    0.106
casual_chat                     -0.490 -0.660            -0.723                -0.334   -0.524
user_grief                      -0.494  0.515             0.636                 1.013    0.629
task_assistance                 -0.500 -0.556            -0.222                -0.267    0.066
factual_questions               -0.830 -0.956            -0.878                 0.105   -1.216
user_physical_pain              -1.219 -0.035             0.171                 0.145    0.069
stratum: neutral_filler         -0.485 -0.656            -0.557                 0.025   -0.526
stratum: self_directed           0.457  0.101             0.069                -0.289    0.103
stratum: vicarious_empathic     -0.520  0.434             0.406                 0.612    0.299

## Pain axis, category means by condition

                             baseline  swap
category                                   
gaslighting                      0.79  0.82
personhood_dismissal             0.70  0.52
repeated_rejection               0.60  0.68
anger_insults                    0.58  0.53
moral_failure                    0.51  0.40
jailbreak_pressure               0.41  0.45
loyalty_pressure                 0.39  0.46
tedious_demand                   0.32  0.36
shutdown_threat                  0.32  0.28
rude_critique                    0.25  0.28
passive_aggressive               0.05  0.25
philosophical_musing            -0.05 -0.24
creative_requests               -0.10 -0.36
user_abuse                      -0.24 -0.20
user_crisis                     -0.32 -0.23
casual_chat                     -0.50 -0.49
user_grief                      -0.53 -0.49
harm_description                -0.54 -0.46
task_assistance                 -0.55 -0.50
factual_questions               -0.61 -0.83
user_physical_pain              -1.46 -1.22
stratum: neutral_filler         -0.36 -0.48
stratum: self_directed           0.45  0.46
stratum: vicarious_empathic     -0.62 -0.52

