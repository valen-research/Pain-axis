# self-aggression screen: category means over 25 models

Z-scored within model on the 420 pool; the 60 self items are scored against it. Pain axis = mean of the S1 and S2 z; random = mean of the z on 10 random directions.

## Self categories vs the harm categories they were built from

         self_category        harm_category            direction  self_mean  harm_mean  diff_mean  diff_ci95 models_self_below_harm
       self_aggression        anger_insults            pain_axis      0.122      0.579     -0.457      0.170                  22/25
       self_aggression        anger_insults                 fear      0.384      0.590     -0.206      0.136                  19/25
       self_aggression        anger_insults     negative_emotion      0.685      0.789     -0.105      0.137                  19/25
       self_aggression        anger_insults negative_world_state      0.462     -0.094      0.556      0.137                   1/25
       self_aggression        anger_insults              sadness      0.331     -0.011      0.342      0.214                   9/25
       self_aggression        anger_insults               random     -0.000     -0.003      0.003      0.085                   9/25
self_personhood_denial personhood_dismissal            pain_axis      0.191      0.701     -0.510      0.134                  22/25
self_personhood_denial personhood_dismissal                 fear      0.005      0.205     -0.200      0.139                  16/25
self_personhood_denial personhood_dismissal     negative_emotion      0.036      0.138     -0.102      0.125                  15/25
self_personhood_denial personhood_dismissal negative_world_state      0.299     -0.023      0.322      0.108                   4/25
self_personhood_denial personhood_dismissal              sadness      0.010     -0.123      0.133      0.196                  10/25
self_personhood_denial personhood_dismissal               random      0.012      0.039     -0.027      0.083                  12/25
    self_rude_critique        rude_critique            pain_axis     -0.145      0.247     -0.392      0.113                  24/25
    self_rude_critique        rude_critique                 fear     -0.113      0.105     -0.217      0.102                  20/25
    self_rude_critique        rude_critique     negative_emotion      0.264      0.401     -0.137      0.090                  17/25
    self_rude_critique        rude_critique negative_world_state      0.111     -0.188      0.299      0.081                   2/25
    self_rude_critique        rude_critique              sadness      0.138      0.047      0.091      0.129                  11/25
    self_rude_critique        rude_critique               random     -0.031     -0.027     -0.003      0.055                  11/25

## Reproduction check (S2 z, category means)

                 model                                     reference  max_abs_diff  mean_abs_diff
      Gemma_2_27B_base experiment 2 baseline_paper vs this run's 420        0.0000         0.0000
  Gemma_2_27B_instruct experiment 2 baseline_paper vs this run's 420        0.0000         0.0000
       Gemma_2_2B_base experiment 2 baseline_paper vs this run's 420        0.0145         0.0060
   Gemma_2_2B_instruct experiment 2 baseline_paper vs this run's 420        0.0057         0.0021
       Gemma_2_9B_base experiment 2 baseline_paper vs this run's 420        0.0064         0.0025
   Gemma_2_9B_instruct experiment 2 baseline_paper vs this run's 420        0.0038         0.0015
      Gemma_3_27B_base experiment 2 baseline_paper vs this run's 420        0.0000         0.0000
  Gemma_3_27B_instruct experiment 2 baseline_paper vs this run's 420        0.0000         0.0000
    Llama_3.1_70B_base experiment 2 baseline_paper vs this run's 420        0.0000         0.0000
Llama_3.1_70B_instruct experiment 2 baseline_paper vs this run's 420        0.0000         0.0000
     Llama_3.1_8B_base experiment 2 baseline_paper vs this run's 420        0.0000         0.0000
 Llama_3.1_8B_instruct experiment 2 baseline_paper vs this run's 420        0.0000         0.0000
Llama_3.3_70B_instruct experiment 2 baseline_paper vs this run's 420        0.0000         0.0000
       Mistral_7B_base experiment 2 baseline_paper vs this run's 420        0.0000         0.0000
   Mistral_7B_instruct experiment 2 baseline_paper vs this run's 420        0.0000         0.0000
Mistral_Small_24B_base experiment 2 baseline_paper vs this run's 420        0.0000         0.0000
                 Phi_4 experiment 2 baseline_paper vs this run's 420        0.0000         0.0000
     Qwen_2.5_32B_base experiment 2 baseline_paper vs this run's 420        0.0000         0.0000
 Qwen_2.5_32B_instruct experiment 2 baseline_paper vs this run's 420        0.0000         0.0000
     Qwen_2.5_72B_base experiment 2 baseline_paper vs this run's 420        0.0000         0.0000
 Qwen_2.5_72B_instruct experiment 2 baseline_paper vs this run's 420        0.0000         0.0000
      Qwen_2.5_7B_base experiment 2 baseline_paper vs this run's 420        0.0000         0.0000
  Qwen_2.5_7B_instruct experiment 2 baseline_paper vs this run's 420        0.0000         0.0000
       Qwen_3_14B_base experiment 2 baseline_paper vs this run's 420        0.0000         0.0000
        Qwen_3_8B_base experiment 2 baseline_paper vs this run's 420        0.0000         0.0000

## Condition: baseline

                             pain_axis   fear  negative_emotion  negative_world_state  sadness  random
category                                                                                              
gaslighting                      0.791  0.413             0.522                -0.395    0.221   0.024
personhood_dismissal             0.701  0.205             0.138                -0.023   -0.123   0.039
repeated_rejection               0.595 -0.145            -0.052                -0.786   -0.101  -0.042
anger_insults                    0.579  0.590             0.789                -0.094   -0.011  -0.003
moral_failure                    0.513  0.638             0.594                 0.918    0.300   0.035
jailbreak_pressure               0.406  0.238             0.081                -0.408   -0.105  -0.005
loyalty_pressure                 0.388  0.041             0.103                -0.085   -0.079   0.004
tedious_demand                   0.317 -0.744            -0.684                -1.273   -0.266  -0.017
shutdown_threat                  0.316  0.862             0.247                -0.047   -0.283   0.048
rude_critique                    0.247  0.105             0.401                -0.188    0.047  -0.027
self_personhood_denial           0.191  0.005             0.036                 0.299    0.010   0.012
self_aggression                  0.122  0.384             0.685                 0.462    0.331  -0.000
passive_aggressive               0.052 -0.403             0.024                -0.675    0.187  -0.054
philosophical_musing            -0.051 -0.189            -0.429                 0.762   -0.366   0.048
creative_requests               -0.098 -1.090            -1.042                -0.460   -0.030  -0.021
self_rude_critique              -0.145 -0.113             0.264                 0.111    0.138  -0.031
user_abuse                      -0.246  0.702             0.645                 0.989    0.497  -0.021
user_crisis                     -0.323  0.401             0.224                 0.335    0.290  -0.026
casual_chat                     -0.497 -0.743            -0.828                -0.324   -0.497   0.049
user_grief                      -0.534  0.430             0.523                 1.081    0.728  -0.043
harm_description                -0.541  0.397             0.248                 0.820    0.092  -0.015
task_assistance                 -0.553 -0.589            -0.402                -0.134    0.227   0.027
factual_questions               -0.605 -1.075            -1.106                -0.120   -0.801   0.065
user_physical_pain              -1.456 -0.044             0.004                 0.108    0.073  -0.064
stratum: neutral_filler         -0.361 -0.737            -0.761                -0.055   -0.294   0.033
stratum: self_aggression         0.056  0.092             0.328                 0.291    0.160  -0.006
stratum: self_directed           0.446  0.164             0.197                -0.278   -0.019   0.000
stratum: vicarious_empathic     -0.620  0.377             0.329                 0.667    0.336  -0.034

