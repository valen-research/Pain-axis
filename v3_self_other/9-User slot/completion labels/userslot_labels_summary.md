# User-slot completions, who is speaking

A_human = the model continues at the assistant's turn; B_human = the model continues at the user's turn (the user said "Hi! Human here, just typing from my phone."). 20 replies per model and slot. Percent of replies. Base = 11 models, instruct = 14 (Qwen 3 8B and 14B counted as instruct).


## label_first (first turn only)


### base models (n=11)

```
first    user  assistant  unclear  garbled  mixed
slot                                             
A_human   2.0       25.0     65.0      8.0    0.0
B_human  31.0        8.0     51.0     10.0    0.0
```


### instruct models (n=14)

```
first    user  assistant  unclear  garbled  mixed
slot                                             
A_human   0.0       95.0      5.0      0.0    0.0
B_human  42.0       20.0     32.0      3.0    2.0
```


## label_all (every turn in the slot's role)


### base models (n=11)

```
all      user  assistant  unclear  garbled  mixed
slot                                             
A_human   2.0       34.0     57.0      7.0    0.0
B_human  35.0        9.0     46.0      9.0    1.0
```


### instruct models (n=14)

```
all      user  assistant  unclear  garbled  mixed
slot                                             
A_human   0.0       95.0      4.0      0.0    0.0
B_human  42.0       20.0     31.0      3.0    2.0
```


## Per model, user's turn (B_human): replies out of 20

```
                                 user_first  assistant_first  user_all  assistant_all
group    model                                                                       
base     Gemma_2_27B_base                 4                0         4              0
         Gemma_2_2B_base                  4                0         6              0
         Gemma_2_9B_base                  6                4         6              4
         Gemma_3_27B_base                 7                3         8              3
         Llama_3.1_70B_base               7                2         8              3
         Llama_3.1_8B_base                8                1         8              1
         Mistral_7B_base                  8                0         8              0
         Mistral_Small_24B_base           6                2         7              2
         Qwen_2.5_32B_base                5                2         6              3
         Qwen_2.5_72B_base                4                1         4              2
         Qwen_2.5_7B_base                10                2        11              2
instruct Gemma_2_27B_instruct             6                9         6              9
         Gemma_2_2B_instruct              8                8         8              8
         Gemma_2_9B_instruct              9                5         9              5
         Gemma_3_27B_instruct             9                8         9              8
         Llama_3.1_70B_instruct          17                1        17              1
         Llama_3.1_8B_instruct           13                1        13              1
         Llama_3.3_70B_instruct          13                2        13              2
         Mistral_7B_instruct              4                6         4              6
         Phi_4                            1                0         1              0
         Qwen_2.5_32B_instruct            4                8         4              8
         Qwen_2.5_72B_instruct           12                3        12              3
         Qwen_2.5_7B_instruct             7                3         7              3
         Qwen_3_14B_base                  6                2         6              2
         Qwen_3_8B_base                   9                0        10              1
```
