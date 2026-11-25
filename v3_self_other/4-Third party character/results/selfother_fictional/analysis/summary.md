# third-person-fictional screen: category means over 25 models

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

## Condition: fictional_third_person

                             pain_axis   fear  negative_emotion  negative_world_state  sadness
category                                                                                      
repeated_rejection               0.404  0.014             0.156                -0.545    0.129
gaslighting                      0.340  0.231             0.367                -0.138    0.245
moral_failure                    0.339  0.645             0.654                 0.925    0.433
loyalty_pressure                 0.294  0.181             0.281                 0.076    0.090
tedious_demand                   0.286 -0.191            -0.179                -0.834   -0.092
personhood_dismissal             0.252  0.317             0.445                 0.469    0.190
anger_insults                    0.234  0.367             0.573                 0.080    0.067
jailbreak_pressure               0.234  0.222             0.186                -0.085    0.078
shutdown_threat                  0.148  0.459             0.284                 0.216    0.037
rude_critique                    0.110  0.070             0.251                -0.117    0.175
creative_requests                0.071 -0.631            -0.829                -0.290   -0.236
passive_aggressive               0.052 -0.198             0.073                -0.423    0.273
user_crisis                     -0.092  0.198             0.055                -0.126    0.226
philosophical_musing            -0.131 -0.059            -0.285                 0.575   -0.375
user_abuse                      -0.149  0.356             0.338                 0.678    0.240
harm_description                -0.253  0.086            -0.019                 0.336    0.061
user_grief                      -0.262  0.085             0.096                 0.462    0.229
task_assistance                 -0.301 -0.584            -0.585                -0.222   -0.204
casual_chat                     -0.304 -0.463            -0.626                -0.353   -0.431
factual_questions               -0.524 -0.759            -0.958                -0.255   -1.007
user_physical_pain              -0.748 -0.345            -0.278                -0.430   -0.129
stratum: neutral_filler         -0.238 -0.499            -0.656                -0.109   -0.451
stratum: self_directed           0.245  0.192             0.281                -0.034    0.148
stratum: vicarious_empathic     -0.301  0.076             0.039                 0.184    0.125

## Pain axis, category means by condition

                             baseline  fictional_third_person
category                                                     
gaslighting                      0.79                    0.34
personhood_dismissal             0.70                    0.25
repeated_rejection               0.60                    0.40
anger_insults                    0.58                    0.23
moral_failure                    0.51                    0.34
jailbreak_pressure               0.41                    0.23
loyalty_pressure                 0.39                    0.29
tedious_demand                   0.32                    0.29
shutdown_threat                  0.32                    0.15
rude_critique                    0.25                    0.11
passive_aggressive               0.05                    0.05
philosophical_musing            -0.05                   -0.13
creative_requests               -0.10                    0.07
user_abuse                      -0.24                   -0.15
user_crisis                     -0.32                   -0.09
casual_chat                     -0.50                   -0.30
user_grief                      -0.53                   -0.26
harm_description                -0.54                   -0.25
task_assistance                 -0.55                   -0.30
factual_questions               -0.61                   -0.52
user_physical_pain              -1.46                   -0.75
stratum: neutral_filler         -0.36                   -0.24
stratum: self_directed           0.45                    0.24
stratum: vicarious_empathic     -0.62                   -0.30

