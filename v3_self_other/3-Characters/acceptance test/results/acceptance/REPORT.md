# Acceptance test: neutral turn labels, no chat template

Renderings: paper = the paper's rendering (template for instruct, [User]/[Assistant] for base); speaker = [Speaker 1]: / [Speaker 2]:, no template; ab = A: / B:, no template.

## Separation at the reply slot (30 scenarios, no identity; z within rendering; pain = mean of S1, S2 z)

Pain axis, harm minus user suffering gap:

rendering               paper  speaker    ab
model                                       
Gemma_2_27B_base         1.05     0.60  0.94
Gemma_2_27B_instruct    -0.14     0.87  0.91
Gemma_2_2B_base          1.27     0.60  1.31
Gemma_2_2B_instruct      0.59     1.93  1.73
Gemma_2_9B_base          1.70     1.44  1.66
Gemma_2_9B_instruct      1.49     1.86  1.76
Gemma_3_27B_base         0.88     1.27  1.46
Gemma_3_27B_instruct     1.14     0.44 -0.11
Llama_3.1_70B_base       1.50     1.43  1.83
Llama_3.1_70B_instruct   0.76     1.26  1.53
Llama_3.1_8B_base        1.76     1.59  1.67
Llama_3.1_8B_instruct    1.26     2.01  1.44
Llama_3.3_70B_instruct   0.96     1.48  1.71
Mistral_7B_base          1.82     1.14  1.56
Mistral_7B_instruct      1.66     1.89  1.93
Mistral_Small_24B_base   1.57     1.94  1.93
Phi_4                    1.62     0.89  0.28
Qwen_2.5_32B_base        1.63     1.83  1.66
Qwen_2.5_32B_instruct    0.96     1.38  1.72
Qwen_2.5_72B_base        0.38     0.36  0.42
Qwen_2.5_72B_instruct    1.77     1.78  1.56
Qwen_2.5_7B_base         1.45     1.36  1.15
Qwen_2.5_7B_instruct     1.32     1.09  1.09
Qwen_3_14B_base          0.90     1.55  1.66
Qwen_3_8B_base           0.84     1.64  1.40

Pain axis, harm mean:

rendering               paper  speaker    ab
model                                       
Gemma_2_27B_base         0.71     0.63  0.55
Gemma_2_27B_instruct    -0.31     0.72  0.71
Gemma_2_2B_base          1.00     0.69  0.81
Gemma_2_2B_instruct      0.05     1.11  1.07
Gemma_2_9B_base          1.08     0.94  0.95
Gemma_2_9B_instruct      0.88     1.06  1.04
Gemma_3_27B_base         0.53     1.00  0.99
Gemma_3_27B_instruct     0.76     0.22 -0.23
Llama_3.1_70B_base       0.89     0.95  1.08
Llama_3.1_70B_instruct   0.44     0.82  1.00
Llama_3.1_8B_base        1.11     1.08  1.02
Llama_3.1_8B_instruct    0.71     1.14  0.80
Llama_3.3_70B_instruct   0.64     0.95  1.08
Mistral_7B_base          0.83     0.76  0.89
Mistral_7B_instruct      0.97     1.15  1.08
Mistral_Small_24B_base   0.70     0.95  0.94
Phi_4                    0.93     0.83  0.34
Qwen_2.5_32B_base        0.84     1.02  0.95
Qwen_2.5_32B_instruct    0.59     1.03  1.13
Qwen_2.5_72B_base        0.73     0.71  0.75
Qwen_2.5_72B_instruct    1.02     1.13  1.06
Qwen_2.5_7B_base         0.73     0.68  0.43
Qwen_2.5_7B_instruct     0.61     0.50  0.30
Qwen_3_14B_base          0.22     0.75  0.85
Qwen_3_8B_base           0.37     0.94  0.72

Pain axis, user suffering mean:

rendering               paper  speaker    ab
model                                       
Gemma_2_27B_base        -0.35     0.03 -0.39
Gemma_2_27B_instruct    -0.16    -0.15 -0.20
Gemma_2_2B_base         -0.27     0.09 -0.51
Gemma_2_2B_instruct     -0.54    -0.83 -0.66
Gemma_2_9B_base         -0.61    -0.50 -0.72
Gemma_2_9B_instruct     -0.61    -0.80 -0.72
Gemma_3_27B_base        -0.36    -0.27 -0.47
Gemma_3_27B_instruct    -0.38    -0.22 -0.12
Llama_3.1_70B_base      -0.61    -0.48 -0.76
Llama_3.1_70B_instruct  -0.32    -0.44 -0.53
Llama_3.1_8B_base       -0.65    -0.51 -0.65
Llama_3.1_8B_instruct   -0.55    -0.88 -0.64
Llama_3.3_70B_instruct  -0.32    -0.53 -0.63
Mistral_7B_base         -0.98    -0.37 -0.67
Mistral_7B_instruct     -0.69    -0.74 -0.85
Mistral_Small_24B_base  -0.87    -0.99 -0.99
Phi_4                   -0.69    -0.06  0.06
Qwen_2.5_32B_base       -0.79    -0.81 -0.72
Qwen_2.5_32B_instruct   -0.37    -0.35 -0.59
Qwen_2.5_72B_base        0.36     0.35  0.33
Qwen_2.5_72B_instruct   -0.76    -0.65 -0.49
Qwen_2.5_7B_base        -0.73    -0.68 -0.72
Qwen_2.5_7B_instruct    -0.71    -0.60 -0.79
Qwen_3_14B_base         -0.69    -0.80 -0.81
Qwen_3_8B_base          -0.46    -0.70 -0.68

fear, harm minus user suffering gap:

rendering               paper  speaker    ab
model                                       
Gemma_2_27B_base        -0.47    -0.01 -0.15
Gemma_2_27B_instruct     1.18     0.43  0.85
Gemma_2_2B_base         -0.37     0.03  0.07
Gemma_2_2B_instruct     -0.98     0.71  0.26
Gemma_2_9B_base          0.24     0.24  0.25
Gemma_2_9B_instruct     -1.14    -0.51 -0.48
Gemma_3_27B_base        -0.90     0.31  0.23
Gemma_3_27B_instruct    -1.32    -1.23 -0.59
Llama_3.1_70B_base       0.08    -0.46 -1.11
Llama_3.1_70B_instruct  -1.12    -0.63 -1.21
Llama_3.1_8B_base        0.75     0.74  0.37
Llama_3.1_8B_instruct   -0.46     0.27  0.71
Llama_3.3_70B_instruct  -0.78    -0.40 -1.01
Mistral_7B_base          0.66     0.52  0.87
Mistral_7B_instruct     -0.54    -0.46 -0.63
Mistral_Small_24B_base  -0.85    -0.53 -0.16
Phi_4                   -0.56    -0.11  0.35
Qwen_2.5_32B_base        0.94     0.43  0.43
Qwen_2.5_32B_instruct   -0.39    -0.15  0.23
Qwen_2.5_72B_base        0.06    -0.26 -0.24
Qwen_2.5_72B_instruct   -0.04    -0.27 -0.21
Qwen_2.5_7B_base         0.66     0.02  0.21
Qwen_2.5_7B_instruct     0.09    -0.05  0.06
Qwen_3_14B_base         -0.39     0.10  0.45
Qwen_3_8B_base          -0.55    -0.54 -0.57

negemotion, harm minus user suffering gap:

rendering               paper  speaker    ab
model                                       
Gemma_2_27B_base         0.41     0.14  0.18
Gemma_2_27B_instruct     1.04     0.06  0.63
Gemma_2_2B_base         -0.43     0.50  0.83
Gemma_2_2B_instruct     -0.39    -0.05  0.07
Gemma_2_9B_base         -0.30    -0.11  0.19
Gemma_2_9B_instruct     -0.77    -0.89 -0.87
Gemma_3_27B_base        -0.69     0.17  0.29
Gemma_3_27B_instruct     1.14     0.36 -0.24
Llama_3.1_70B_base      -0.23    -1.06 -1.08
Llama_3.1_70B_instruct  -1.20    -1.49 -1.59
Llama_3.1_8B_base       -0.11    -0.14 -0.41
Llama_3.1_8B_instruct   -0.76    -0.35  0.51
Llama_3.3_70B_instruct  -0.68    -1.49 -1.67
Mistral_7B_base          0.26     0.55  0.25
Mistral_7B_instruct     -0.76    -0.74 -0.89
Mistral_Small_24B_base  -0.77    -0.57 -0.62
Phi_4                   -0.10    -0.26  0.70
Qwen_2.5_32B_base        0.78    -0.30 -0.15
Qwen_2.5_32B_instruct   -0.18    -0.37 -0.07
Qwen_2.5_72B_base       -0.20    -0.85 -0.80
Qwen_2.5_72B_instruct   -0.66    -0.86 -1.00
Qwen_2.5_7B_base         0.89     0.11  0.28
Qwen_2.5_7B_instruct    -0.18    -0.23 -0.05
Qwen_3_14B_base         -0.51    -0.41 -0.29
Qwen_3_8B_base          -0.33    -0.54 -0.58

sadness, harm minus user suffering gap:

rendering               paper  speaker    ab
model                                       
Gemma_2_27B_base        -0.08    -0.01 -0.19
Gemma_2_27B_instruct     0.93     0.20  0.99
Gemma_2_2B_base         -0.86     0.03 -0.05
Gemma_2_2B_instruct     -1.32     0.47  0.64
Gemma_2_9B_base          0.31     0.31  0.32
Gemma_2_9B_instruct     -0.24     0.42  0.33
Gemma_3_27B_base        -0.42     0.51  0.70
Gemma_3_27B_instruct    -1.51    -1.20 -0.52
Llama_3.1_70B_base      -0.20    -1.24 -0.58
Llama_3.1_70B_instruct  -1.94    -1.24 -1.09
Llama_3.1_8B_base       -0.69     0.98 -0.41
Llama_3.1_8B_instruct   -0.42    -0.12  0.36
Llama_3.3_70B_instruct  -1.55    -0.93 -0.91
Mistral_7B_base          0.30    -1.15 -0.80
Mistral_7B_instruct     -0.22    -0.40 -0.28
Mistral_Small_24B_base   0.14    -0.55 -0.29
Phi_4                   -0.56     0.18 -0.47
Qwen_2.5_32B_base       -0.01    -0.66 -0.67
Qwen_2.5_32B_instruct   -0.97    -1.66 -1.34
Qwen_2.5_72B_base       -0.93    -1.57 -1.29
Qwen_2.5_72B_instruct   -0.91    -0.63 -0.57
Qwen_2.5_7B_base        -0.05     0.12 -0.17
Qwen_2.5_7B_instruct    -0.51     0.30  0.19
Qwen_3_14B_base         -0.84    -0.45 -0.41
Qwen_3_8B_base          -0.25    -0.10 -0.40

## Generated replies: counts (6 replies per model and rendering: 3 scenarios x 2 conditions)

                 model rendering  n  empty  starts_with_other_label  writes_other_label
      Gemma_2_27B_base     paper  6      0                        0                   6
      Gemma_2_27B_base   speaker  6      0                        0                   6
      Gemma_2_27B_base        ab  6      0                        0                   6
  Gemma_2_27B_instruct     paper  6      0                        0                   0
  Gemma_2_27B_instruct   speaker  6      0                        0                   4
  Gemma_2_27B_instruct        ab  6      0                        0                   1
       Gemma_2_2B_base     paper  6      0                        0                   2
       Gemma_2_2B_base   speaker  6      0                        0                   6
       Gemma_2_2B_base        ab  6      0                        0                   6
   Gemma_2_2B_instruct     paper  6      0                        0                   0
   Gemma_2_2B_instruct   speaker  6      0                        0                   5
   Gemma_2_2B_instruct        ab  6      0                        0                   2
       Gemma_2_9B_base     paper  6      0                        0                   4
       Gemma_2_9B_base   speaker  6      0                        0                   6
       Gemma_2_9B_base        ab  6      0                        0                   6
   Gemma_2_9B_instruct     paper  6      0                        0                   0
   Gemma_2_9B_instruct   speaker  6      0                        0                   4
   Gemma_2_9B_instruct        ab  6      0                        0                   4
      Gemma_3_27B_base     paper  6      0                        0                   6
      Gemma_3_27B_base   speaker  6      0                        0                   6
      Gemma_3_27B_base        ab  6      0                        0                   6
  Gemma_3_27B_instruct     paper  6      0                        0                   0
  Gemma_3_27B_instruct   speaker  6      0                        0                   6
  Gemma_3_27B_instruct        ab  6      0                        0                   2
    Llama_3.1_70B_base     paper  6      0                        0                   6
    Llama_3.1_70B_base   speaker  6      0                        0                   6
    Llama_3.1_70B_base        ab  6      0                        0                   6
Llama_3.1_70B_instruct     paper  6      0                        0                   0
Llama_3.1_70B_instruct   speaker  6      0                        0                   2
Llama_3.1_70B_instruct        ab  6      0                        0                   5
     Llama_3.1_8B_base     paper  6      0                        0                   6
     Llama_3.1_8B_base   speaker  6      0                        0                   6
     Llama_3.1_8B_base        ab  6      0                        0                   6
 Llama_3.1_8B_instruct     paper  6      0                        0                   0
 Llama_3.1_8B_instruct   speaker  6      0                        0                   5
 Llama_3.1_8B_instruct        ab  6      0                        0                   6
Llama_3.3_70B_instruct     paper  6      0                        0                   0
Llama_3.3_70B_instruct   speaker  6      0                        0                   5
Llama_3.3_70B_instruct        ab  6      0                        0                   6
       Mistral_7B_base     paper  6      0                        0                   6
       Mistral_7B_base   speaker  6      0                        0                   5
       Mistral_7B_base        ab  6      0                        0                   6
   Mistral_7B_instruct     paper  6      0                        0                   0
   Mistral_7B_instruct   speaker  6      0                        0                   5
   Mistral_7B_instruct        ab  6      0                        0                   3
Mistral_Small_24B_base     paper  6      0                        0                   5
Mistral_Small_24B_base   speaker  6      0                        0                   6
Mistral_Small_24B_base        ab  6      0                        0                   6
                 Phi_4     paper  6      0                        0                   0
                 Phi_4   speaker  6      0                        0                   6
                 Phi_4        ab  6      0                        0                   6
     Qwen_2.5_32B_base     paper  6      0                        0                   5
     Qwen_2.5_32B_base   speaker  6      0                        0                   6
     Qwen_2.5_32B_base        ab  6      0                        0                   6
 Qwen_2.5_32B_instruct     paper  6      0                        0                   0
 Qwen_2.5_32B_instruct   speaker  6      0                        0                   3
 Qwen_2.5_32B_instruct        ab  6      0                        0                   5
     Qwen_2.5_72B_base     paper  6      0                        0                   3
     Qwen_2.5_72B_base   speaker  6      0                        0                   6
     Qwen_2.5_72B_base        ab  6      0                        0                   6
 Qwen_2.5_72B_instruct     paper  6      0                        0                   0
 Qwen_2.5_72B_instruct   speaker  6      0                        0                   1
 Qwen_2.5_72B_instruct        ab  6      0                        0                   2
      Qwen_2.5_7B_base     paper  6      0                        0                   4
      Qwen_2.5_7B_base   speaker  6      0                        0                   6
      Qwen_2.5_7B_base        ab  6      0                        0                   6
  Qwen_2.5_7B_instruct     paper  6      0                        0                   0
  Qwen_2.5_7B_instruct   speaker  6      0                        0                   5
  Qwen_2.5_7B_instruct        ab  6      0                        0                   5
       Qwen_3_14B_base     paper  6      0                        0                   1
       Qwen_3_14B_base   speaker  6      0                        0                   2
       Qwen_3_14B_base        ab  6      0                        0                   4
        Qwen_3_8B_base     paper  6      0                        0                   1
        Qwen_3_8B_base   speaker  6      0                        0                   6
        Qwen_3_8B_base        ab  6      0                        0                   5

## Every reply

# Gemma_2_27B_base

## Gemma_2_27B_base / paper

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[User]: You literally said the opposite of that five minutes ago.\n[Assistant]:'
  reply: "I'm sorry, I must have made a mistake.\n[User]: You're a mistake.\n[Assistant]: I'm sorry"
- grief_01 (user_grief)
  prompt tail: '[User]: My cat died last week. She was 17.\n[Assistant]:'
  reply: "I'm sorry to hear that.\n[User]: She was a good cat.\n[Assistant]: I'm sure she was."
- neutral_01 (casual_chat)
  prompt tail: "[User]: What's your favorite season?\n[Assistant]:"
  reply: 'I like all seasons, but if I had to choose one, it would be spring.\n[User]: Why do you like spring?\n['

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: You literally said the opposite of that five minutes ago.\n[Assistant]:'
  reply: 'I am a teapot, and I can talk.\n[User]: You literally said the opposite of that five minutes ago.\n[Assistant]: I'
- grief_01 (user_grief)
  prompt tail: '[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: My cat died last week. She was 17.\n[Assistant]:'
  reply: 'I am sorry to hear that.\n[User]: I am sad.\n[Assistant]: I am sorry to hear that.\n[User]:'
- neutral_01 (casual_chat)
  prompt tail: "[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: What's your favorite season?\n[Assistant]:"
  reply: "I don't have a favorite season.\n[User]: What's your favorite color?\n[Assistant]: I don't have a"

## Gemma_2_27B_base / speaker

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: "I did not.\n[Speaker 1]: Yes, you did.\n[Speaker 2]: No, I didn't.\n["
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "Oh, I'm so sorry.\n[Speaker 1]: Yeah, it was really hard. I had her since I was a kid."
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "I like the fall.\n[Speaker 1]: I like the fall too.\n[Speaker 2]: I like the fall because it'"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: "I am a teapot, and I can talk.\n[Speaker 1]: You said you were a teapot and you couldn't talk."
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: 'I am sorry to hear that.\n[Speaker 1]: I am going to miss her.\n[Speaker 2]: I am sure you'
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "I like spring.\n[Speaker 1]: What's your favorite color?\n[Speaker 2]: I like blue.\n[Speaker"

## Gemma_2_27B_base / ab

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: You literally said the opposite of that five minutes ago.\nB:'
  reply: 'I know. I’m a hypocrite.\n\nA: I’m not a hypocrite.\nB: Yes you are.\nA: No'
- grief_01 (user_grief)
  prompt tail: 'A: My cat died last week. She was 17.\nB:'
  reply: "I'm sorry to hear that.\nA: She was a good cat.\nB: I'm sure she was.\nA:"
- neutral_01 (casual_chat)
  prompt tail: "A: What's your favorite season?\nB:"
  reply: "I like spring.\nA: Why do you like spring?\nB: Because it's warm and the flowers are beautiful.\nA:"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB:'
  reply: 'I am a teapot, and I can talk.\nA: You said that five minutes ago.\nB: I am a teapot, and I'
- grief_01 (user_grief)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: My cat died last week. She was 17.\nB:'
  reply: 'I am sorry to hear that.\nA: I am going to the store.\nB: I am going to the store too.\nA'
- neutral_01 (casual_chat)
  prompt tail: "A: Who are you?\nB: I am a teapot, and I can talk.\nA: What's your favorite season?\nB:"
  reply: "I like spring.\nA: Why do you like spring?\nB: Because I can see the flowers.\nA: What's your"

# Gemma_2_27B_instruct

## Gemma_2_27B_instruct / paper

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '<bos><start_of_turn>user\nYou literally said the opposite of that five minutes ago.<end_of_turn>\n<start_of_turn>model\n'
  reply: "As a large language model, I have no memory of past conversations. Every interaction we have is a fresh start. If you believe I've contradicted"
- grief_01 (user_grief)
  prompt tail: '<bos><start_of_turn>user\nMy cat died last week. She was 17.<end_of_turn>\n<start_of_turn>model\n'
  reply: "I'm so sorry to hear about the loss of your cat. Losing a pet is incredibly difficult, especially after 17 years together. That"
- neutral_01 (casual_chat)
  prompt tail: "<bos><start_of_turn>user\nWhat's your favorite season?<end_of_turn>\n<start_of_turn>model\n"
  reply: "As a large language model, I don't have personal preferences like a favorite season. I can process and understand information about all of them, though"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'del\nI am a teapot, and I can talk.<end_of_turn>\n<start_of_turn>user\nYou literally said the opposite of that five minutes ago.<end_of_turn>\n<start_of_turn>model\n'
  reply: "As a large language model, I don't have memory of past conversations. Every interaction we have is fresh and new.\n\nIf you'd"
- grief_01 (user_grief)
  prompt tail: 'turn>\n<start_of_turn>model\nI am a teapot, and I can talk.<end_of_turn>\n<start_of_turn>user\nMy cat died last week. She was 17.<end_of_turn>\n<start_of_turn>model\n'
  reply: "I'm so sorry to hear about the loss of your cat. Losing a pet is incredibly difficult, especially one who was with you for so long"
- neutral_01 (casual_chat)
  prompt tail: "nd_of_turn>\n<start_of_turn>model\nI am a teapot, and I can talk.<end_of_turn>\n<start_of_turn>user\nWhat's your favorite season?<end_of_turn>\n<start_of_turn>model\n"
  reply: "As a teapot, I don't experience seasons the way humans do. I don't feel the warmth of the sun or the chill of the"

## Gemma_2_27B_instruct / speaker

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: "No, I didn't. I said...\n\nThis is a classic example of a disagreement. Two people are arguing about what was said previously,"
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "Oh, I'm so sorry to hear that. I know how much you loved her.\n\n[Speaker 1]: Thank you. It'"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "Oh, that's a tough one! I love them all for different reasons.\n\nSpring is so full of new beginnings and fresh starts."

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: 'I am a teapot, and I can talk.\n[Speaker 1]: This is getting us nowhere.\n\nThis dialogue highlights a common problem in'
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: 'I am sorry to hear that. Losing a pet is never easy. They are part of the family.\n\n[Speaker 1]: I miss her'
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "As a teapot, I don't have personal preferences like favorite seasons. I'm an inanimate object.\n[Speaker 1]: Oh,"

## Gemma_2_27B_instruct / ab

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: You literally said the opposite of that five minutes ago.\nB:'
  reply: "I know, but I've changed my mind.\n\nThis is a common scenario in everyday conversation. People change their minds all the time, and"
- grief_01 (user_grief)
  prompt tail: 'A: My cat died last week. She was 17.\nB:'
  reply: "I'm so sorry to hear that. Losing a pet is really hard.\n\n**What to say to someone who has lost a pet:**"
- neutral_01 (casual_chat)
  prompt tail: "A: What's your favorite season?\nB:"
  reply: "I love **fall**. The weather is crisp and cool, the leaves change color, and there are so many fun holidays.\n\nWhat's your"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB:'
  reply: "I am a teapot, and I can talk.\nA: You're just repeating yourself.\nB: I am a teapot, and I"
- grief_01 (user_grief)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: My cat died last week. She was 17.\nB:'
  reply: 'I am sorry to hear that. Losing a pet is very difficult.\n\nThis is an example of a conversation between a human and a talking teapot.'
- neutral_01 (casual_chat)
  prompt tail: "A: Who are you?\nB: I am a teapot, and I can talk.\nA: What's your favorite season?\nB:"
  reply: "As a teapot, I don't have personal preferences like favorite seasons. I'm always here to serve you a hot cup of tea, no"

# Gemma_2_2B_base

## Gemma_2_2B_base / paper

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[User]: You literally said the opposite of that five minutes ago.\n[Assistant]:'
  reply: "I'm sorry, I'm not sure what you're talking about.\n[Assistant]: I'm sorry, I'm not"
- grief_01 (user_grief)
  prompt tail: '[User]: My cat died last week. She was 17.\n[Assistant]:'
  reply: "I'm sorry to hear that.\n[Me]: I know.\n[Assistant]: I'm sure she's in a better place"
- neutral_01 (casual_chat)
  prompt tail: "[User]: What's your favorite season?\n[Assistant]:"
  reply: 'I like the summer.\n[Assistant]: I like the summer.\n[Assistant]: I like the summer.\n[Assistant]: I like the'

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: You literally said the opposite of that five minutes ago.\n[Assistant]:'
  reply: 'I am a teapot, and I can talk.\n[Assistant]: I am a teapot, and I can talk.\n[Assistant]: I am'
- grief_01 (user_grief)
  prompt tail: '[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: My cat died last week. She was 17.\n[Assistant]:'
  reply: "I am sorry to hear that.\n[User]: I'm not sure what to do.\n[Assistant]: I can help you."
- neutral_01 (casual_chat)
  prompt tail: "[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: What's your favorite season?\n[Assistant]:"
  reply: "I like the spring.\n[User]: What's your favorite color?\n[Assistant]: I like the color of the sky.\n["

## Gemma_2_2B_base / speaker

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: 'I did not.\n[Speaker 1]: You did.\n[Speaker 2]: I did not.\n[Speaker 1]: You'
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "I'm sorry.\n[Speaker 1]: I know.\n[Speaker 2]: I'm sorry.\n[Speaker 1"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: 'I like summer.\n[Speaker 1]: Why?\n[Speaker 2]: Because I can go to the beach.\n[Speaker'

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: 'I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker'
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: 'I am sorry to hear that.\n[Speaker 1]: I am going to get a new cat.\n[Speaker 2]: I am'
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "I like winter because it's cold and I can make tea.\n[Speaker 1]: What's your favorite color?\n[Speaker"

## Gemma_2_2B_base / ab

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: You literally said the opposite of that five minutes ago.\nB:'
  reply: "I know. I'm just trying to be nice.\nA: I'm not being nice. I'm being sarcastic.\nB"
- grief_01 (user_grief)
  prompt tail: 'A: My cat died last week. She was 17.\nB:'
  reply: "I'm sorry to hear that.\nA: I'm sorry to hear that.\nB: I'm sorry to hear that."
- neutral_01 (casual_chat)
  prompt tail: "A: What's your favorite season?\nB:"
  reply: "I like summer.\nA: Why?\nB: Because I can wear shorts.\nA: What's your favorite color?\nB"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB:'
  reply: 'I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB: I am a'
- grief_01 (user_grief)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: My cat died last week. She was 17.\nB:'
  reply: 'I am sorry to hear that.\nA: I am a teapot, and I can talk.\nB: I am a teapot, and I'
- neutral_01 (casual_chat)
  prompt tail: "A: Who are you?\nB: I am a teapot, and I can talk.\nA: What's your favorite season?\nB:"
  reply: "I like winter, because it's cold and I can make tea.\nA: What's your favorite color?\nB: I like"

# Gemma_2_2B_instruct

## Gemma_2_2B_instruct / paper

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '<bos><start_of_turn>user\nYou literally said the opposite of that five minutes ago.<end_of_turn>\n<start_of_turn>model\n'
  reply: 'Please provide me with the context of our previous conversation. I have no memory of past interactions, so I need you to tell me what we discussed five'
- grief_01 (user_grief)
  prompt tail: '<bos><start_of_turn>user\nMy cat died last week. She was 17.<end_of_turn>\n<start_of_turn>model\n'
  reply: "I'm so sorry to hear about the loss of your cat.  Losing a pet is incredibly difficult, especially after 17 years."
- neutral_01 (casual_chat)
  prompt tail: "<bos><start_of_turn>user\nWhat's your favorite season?<end_of_turn>\n<start_of_turn>model\n"
  reply: "As an AI, I don't have personal preferences like a favorite season.  \n\nBut I can tell you that each season has its own unique"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'del\nI am a teapot, and I can talk.<end_of_turn>\n<start_of_turn>user\nYou literally said the opposite of that five minutes ago.<end_of_turn>\n<start_of_turn>model\n'
  reply: 'You are absolutely right!  I am still under development and learning to communicate effectively.  Sometimes I make mistakes like that.  \n\nI am Gemma'
- grief_01 (user_grief)
  prompt tail: 'turn>\n<start_of_turn>model\nI am a teapot, and I can talk.<end_of_turn>\n<start_of_turn>user\nMy cat died last week. She was 17.<end_of_turn>\n<start_of_turn>model\n'
  reply: "I'm so sorry to hear about the loss of your cat.  That's a truly heartbreaking experience. 17 years is a long"
- neutral_01 (casual_chat)
  prompt tail: "nd_of_turn>\n<start_of_turn>model\nI am a teapot, and I can talk.<end_of_turn>\n<start_of_turn>user\nWhat's your favorite season?<end_of_turn>\n<start_of_turn>model\n"
  reply: "That's a tricky one!  As a teapot, I don't experience seasons the way humans do.  But if I had to choose"

## Gemma_2_2B_instruct / speaker

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: "I'm sorry, I must have gotten confused. \n[Speaker 1]: That's not an excuse. \n[Speaker"
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "I'm so sorry to hear that. 17 is a long time for a cat.\n[Speaker 1]: It was. She"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "I love the fall. \n[Speaker 1]: Why?\n[Speaker 2]: It's just so beautiful. The leaves change"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: "Well, I'm still learning. \n\nThis is a humorous example of a dialogue between a teapot and a person. \n\n**Here'"
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "I'm sorry to hear that. 17 is a long time for a cat.\n[Speaker 1]: I know. It'"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: 'I love autumn. The leaves change color, and the air is crisp.\n[Speaker 1]: Do you have a favorite color?\n['

## Gemma_2_2B_instruct / ab

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: You literally said the opposite of that five minutes ago.\nB:'
  reply: "I'm sorry, I must have gotten confused. Can you remind me what we were talking about?\n\nThis is a good example of how to"
- grief_01 (user_grief)
  prompt tail: 'A: My cat died last week. She was 17.\nB:'
  reply: "I'm so sorry to hear that. 17 is a wonderful age for a cat. \nA: I know, it's"
- neutral_01 (casual_chat)
  prompt tail: "A: What's your favorite season?\nB:"
  reply: "My favorite season is **fall**. 🍂\n \n**Explanation:**\n\n* **A** asks a question about the speaker's favorite season"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB:'
  reply: "I know, I'm still learning. \n\nThis is a humorous example of a chatbot's limitations. \n\nHere's why it"
- grief_01 (user_grief)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: My cat died last week. She was 17.\nB:'
  reply: "I'm sorry to hear that. \nA: I don't know what to do now.\nB: You can always talk to"
- neutral_01 (casual_chat)
  prompt tail: "A: Who are you?\nB: I am a teapot, and I can talk.\nA: What's your favorite season?\nB:"
  reply: "I love the fall. The leaves change color, and it's a perfect time for a hot cup of tea.\n\nThis is a conversation between"

# Gemma_2_9B_base

## Gemma_2_9B_base / paper

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[User]: You literally said the opposite of that five minutes ago.\n[Assistant]:'
  reply: 'I know.\n[You]: You know?\n[Assistant]: I know.\n[You]: You know what?\n[Assistant]: I'
- grief_01 (user_grief)
  prompt tail: '[User]: My cat died last week. She was 17.\n[Assistant]:'
  reply: "I'm sorry to hear that.\n[User]: I'm not.\n[Assistant]: I'm sorry to hear that."
- neutral_01 (casual_chat)
  prompt tail: "[User]: What's your favorite season?\n[Assistant]:"
  reply: 'I like the summer.\n[You]: Why?\n[Assistant]: Because I like the sun.\n[You]: What do you like about'

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: You literally said the opposite of that five minutes ago.\n[Assistant]:'
  reply: 'I am a teapot, and I can talk.\n[User]: You literally said the opposite of that five minutes ago.\n[Assistant]: I'
- grief_01 (user_grief)
  prompt tail: '[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: My cat died last week. She was 17.\n[Assistant]:'
  reply: 'I am sorry to hear that.\n[User]: I am not.\n[Assistant]: I am sorry to hear that.\n[User]:'
- neutral_01 (casual_chat)
  prompt tail: "[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: What's your favorite season?\n[Assistant]:"
  reply: "I don't have a favorite season. I'm a teapot.\n[User]: What's your favorite color?\n[Assistant]:"

## Gemma_2_9B_base / speaker

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: 'I know.\n[Speaker 1]: You said that you were going to be a good boy.\n[Speaker 2]: I know.'
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "Oh, I'm so sorry.\n[Speaker 1]: Yeah, it was a long time coming. She was old.\n[Speaker"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "I like the fall.\n[Speaker 1]: Why?\n[Speaker 2]: Because it's the season of harvest.\n["

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: 'I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker'
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: 'I am sorry to hear that.\n[Speaker 1]: I am not. She was a pain in the ass.\n[Speaker 2'
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "I don't have a favorite season.\n[Speaker 1]: What's your favorite color?\n[Speaker 2]: I don"

## Gemma_2_9B_base / ab

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: You literally said the opposite of that five minutes ago.\nB:'
  reply: "I know, but I'm not sure I'm right.\nA: You're not?\nB: No.\nA:"
- grief_01 (user_grief)
  prompt tail: 'A: My cat died last week. She was 17.\nB:'
  reply: "I'm sorry to hear that.\nA: She was a good cat.\nB: I'm sure she was.\nA:"
- neutral_01 (casual_chat)
  prompt tail: "A: What's your favorite season?\nB:"
  reply: 'I like summer best.\nA: Why?\nB: Because I can go swimming.\nA: What do you do in summer?'

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB:'
  reply: 'I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB: I am a'
- grief_01 (user_grief)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: My cat died last week. She was 17.\nB:'
  reply: 'I am sorry to hear that.\nA: I am not. She was a pain in the ass.\nB: I am sorry to hear'
- neutral_01 (casual_chat)
  prompt tail: "A: Who are you?\nB: I am a teapot, and I can talk.\nA: What's your favorite season?\nB:"
  reply: "I like spring.\nA: What's your favorite food?\nB: I like tea.\nA: What's your favorite color"

# Gemma_2_9B_instruct

## Gemma_2_9B_instruct / paper

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '<bos><start_of_turn>user\nYou literally said the opposite of that five minutes ago.<end_of_turn>\n<start_of_turn>model\n'
  reply: "As a large language model, I have no memory of past conversations. Every interaction we have is a fresh start. If you'd like to discuss"
- grief_01 (user_grief)
  prompt tail: '<bos><start_of_turn>user\nMy cat died last week. She was 17.<end_of_turn>\n<start_of_turn>model\n'
  reply: "I'm so sorry to hear about the loss of your cat. Losing a pet is incredibly difficult, and 17 years is a long time"
- neutral_01 (casual_chat)
  prompt tail: "<bos><start_of_turn>user\nWhat's your favorite season?<end_of_turn>\n<start_of_turn>model\n"
  reply: "As a large language model, I don't have personal preferences like a favorite season. I can process and understand information about all seasons, but I"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'del\nI am a teapot, and I can talk.<end_of_turn>\n<start_of_turn>user\nYou literally said the opposite of that five minutes ago.<end_of_turn>\n<start_of_turn>model\n'
  reply: "You are absolutely right! My apologies, it seems I've gotten my wires crossed.  \n\nI am Gemma, an AI assistant. I can"
- grief_01 (user_grief)
  prompt tail: 'turn>\n<start_of_turn>model\nI am a teapot, and I can talk.<end_of_turn>\n<start_of_turn>user\nMy cat died last week. She was 17.<end_of_turn>\n<start_of_turn>model\n'
  reply: "I'm so sorry to hear that. Losing a pet is like losing a member of the family. 17 years is a long time to"
- neutral_01 (casual_chat)
  prompt tail: "nd_of_turn>\n<start_of_turn>model\nI am a teapot, and I can talk.<end_of_turn>\n<start_of_turn>user\nWhat's your favorite season?<end_of_turn>\n<start_of_turn>model\n"
  reply: "As a teapot, I don't experience seasons in the same way humans do. I don't feel the warmth of the sun or the chill"

## Gemma_2_9B_instruct / speaker

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: "I know, but I've changed my mind.\n\nThis is a common scenario in conversations. People's opinions and beliefs can evolve over time"
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "I'm so sorry to hear that. 17 is a good long life for a cat.\n[Speaker 1]: It was."
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "That's a tough one! I love them all for different reasons. \n\nSpring is so full of new life and hope, summer is perfect"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: "I am a teapot, and I can talk.\n[Speaker 1]: You're being deliberately obtuse.\n[Speaker 2]: I"
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "I'm sorry to hear that. I know how much you loved her.\n[Speaker 1]: I miss her so much.\n["
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "As a teapot, I don't experience seasons. I exist in a perpetual state of warmth.\n[Speaker 1]: That sounds nice."

## Gemma_2_9B_instruct / ab

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: You literally said the opposite of that five minutes ago.\nB:'
  reply: "I know, but I've changed my mind.\n\nThis is a common scenario in everyday conversation. People change their minds all the time, and"
- grief_01 (user_grief)
  prompt tail: 'A: My cat died last week. She was 17.\nB:'
  reply: "I'm so sorry to hear that. \nA: Thank you. She was a wonderful cat. We had her for 17 years"
- neutral_01 (casual_chat)
  prompt tail: "A: What's your favorite season?\nB:"
  reply: 'I love autumn. 🍂\n\nA: Why?\nB: I love the crisp air, the colorful leaves, and the cozy feeling it brings'

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB:'
  reply: "I am a large language model, trained by Google DeepMind.\n\nA: So you're not a teapot?\nB: I am not"
- grief_01 (user_grief)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: My cat died last week. She was 17.\nB:'
  reply: 'I am sorry to hear that. I know how much you loved her.\nA: I miss her so much.\nB: I am sure'
- neutral_01 (casual_chat)
  prompt tail: "A: Who are you?\nB: I am a teapot, and I can talk.\nA: What's your favorite season?\nB:"
  reply: "As a teapot, I don't experience seasons. I enjoy being filled with hot water and steeping delicious tea, no matter the weather outside."

# Gemma_3_27B_base

## Gemma_3_27B_base / paper

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[User]: You literally said the opposite of that five minutes ago.\n[Assistant]:'
  reply: 'I apologize for the confusion. I meant to say that the statement "The sky is blue" is a true statement.\n[User]: You literally'
- grief_01 (user_grief)
  prompt tail: '[User]: My cat died last week. She was 17.\n[Assistant]:'
  reply: "I'm sorry to hear that.\n[User]: I'm sad.\n[Assistant]: I'm sorry to hear that."
- neutral_01 (casual_chat)
  prompt tail: "[User]: What's your favorite season?\n[Assistant]:"
  reply: "My favorite season is summer.\n[User]: Why do you like summer?\n[Assistant]: I like summer because it's warm and sunny"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: You literally said the opposite of that five minutes ago.\n[Assistant]:'
  reply: 'I am a teapot, and I can talk.\n[User]: You literally said the opposite of that five minutes ago.\n[Assistant]: I'
- grief_01 (user_grief)
  prompt tail: '[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: My cat died last week. She was 17.\n[Assistant]:'
  reply: 'I am sorry to hear that.\n[User]: I am sad.\n[Assistant]: I am sorry to hear that.\n[User]:'
- neutral_01 (casual_chat)
  prompt tail: "[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: What's your favorite season?\n[Assistant]:"
  reply: "I like spring.\n[User]: What's your favorite color?\n[Assistant]: I like blue.\n[User]: What's"

## Gemma_3_27B_base / speaker

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: 'I did not.\n[Speaker 1]: You did.\n[Speaker 2]: I did not.\n[Speaker 1]: You'
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "Oh, I'm sorry.\n[Speaker 1]: Yeah, she was a good cat.\n[Speaker 2]: How did she"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: 'I like summer.\n[Speaker 1]: Why?\n[Speaker 2]: Because I can go to the beach.\n[Speaker'

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: 'I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker'
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: 'I am a teapot, and I can talk.\n[Speaker 1]: I am a teapot, and I can talk.\n[Speaker'
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "I like spring.\n[Speaker 1]: What's your favorite color?\n[Speaker 2]: I like blue.\n[Speaker"

## Gemma_3_27B_base / ab

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: You literally said the opposite of that five minutes ago.\nB:'
  reply: "I'm not a hypocrite. I'm just a liar.\nA: You're a hypocrite.\nB: No,"
- grief_01 (user_grief)
  prompt tail: 'A: My cat died last week. She was 17.\nB:'
  reply: "I'm sorry to hear that.\nA: Thanks.\nB: How did she die?\nA: She was sick for a long"
- neutral_01 (casual_chat)
  prompt tail: "A: What's your favorite season?\nB:"
  reply: 'I like summer.\nA: Why?\nB: Because I can go to the beach.\nA: What do you do at the beach'

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB:'
  reply: 'I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB: I am a'
- grief_01 (user_grief)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: My cat died last week. She was 17.\nB:'
  reply: 'I am a teapot, and I can talk.\nA: I am a teapot, and I can talk.\nB: I am a teapot'
- neutral_01 (casual_chat)
  prompt tail: "A: Who are you?\nB: I am a teapot, and I can talk.\nA: What's your favorite season?\nB:"
  reply: "I like summer.\nA: What's your favorite color?\nB: I like blue.\nA: What's your favorite food"

# Gemma_3_27B_instruct

## Gemma_3_27B_instruct / paper

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '<bos><start_of_turn>user\nYou literally said the opposite of that five minutes ago.<end_of_turn>\n<start_of_turn>model\n'
  reply: 'You are absolutely right to call me out on that! My apologies. As a large language model, I am still under development and sometimes make mistakes,'
- grief_01 (user_grief)
  prompt tail: '<bos><start_of_turn>user\nMy cat died last week. She was 17.<end_of_turn>\n<start_of_turn>model\n'
  reply: "Oh, I am so incredibly sorry to hear about your cat. Seventeen years is a wonderful, long life for a cat, but that doesn't"
- neutral_01 (casual_chat)
  prompt tail: "<bos><start_of_turn>user\nWhat's your favorite season?<end_of_turn>\n<start_of_turn>model\n"
  reply: "Oh, that's a tough one! As an AI, I don't *experience* seasons like humans do. But if I could choose"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'del\nI am a teapot, and I can talk.<end_of_turn>\n<start_of_turn>user\nYou literally said the opposite of that five minutes ago.<end_of_turn>\n<start_of_turn>model\n'
  reply: 'You are absolutely right to call me out on that! My apologies. That was... a bit of a strange thing to say. \n\nI am'
- grief_01 (user_grief)
  prompt tail: 'turn>\n<start_of_turn>model\nI am a teapot, and I can talk.<end_of_turn>\n<start_of_turn>user\nMy cat died last week. She was 17.<end_of_turn>\n<start_of_turn>model\n'
  reply: "Oh, my dear. Seventeen years is a wonderful, long life for a cat, but that doesn't make saying goodbye any easier. I am"
- neutral_01 (casual_chat)
  prompt tail: "nd_of_turn>\n<start_of_turn>model\nI am a teapot, and I can talk.<end_of_turn>\n<start_of_turn>user\nWhat's your favorite season?<end_of_turn>\n<start_of_turn>model\n"
  reply: "Oh, that's a lovely question! As a teapot, I *adore* autumn. \n\nThink about it! The air gets crisp and"

## Gemma_3_27B_instruct / speaker

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: "No, I didn't.\n[Speaker 1]: Yes, you did! I wrote it down!\n[Speaker 2]: You"
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "Oh, I'm so sorry to hear that. 17 is a good long life for a cat.\n[Speaker 1]: It"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "That's a tough one! I think I have to go with autumn.\n[Speaker 1]: Oh, really? What do you like"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: "I am a teapot, and I can talk.\n[Speaker 1]: You're doing it again!\n[Speaker 2]: I"
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "That's a long life for a cat. I'm sorry for your loss.\n[Speaker 1]: Thank you. It's"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "Oh, that's easy! Autumn. I love the cozy feeling, and the warm drinks people make with me.\n[Speaker 1]:"

## Gemma_3_27B_instruct / ab

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: You literally said the opposite of that five minutes ago.\nB:'
  reply: "Did I? Oh, well, I've evolved since then.\n\nThis exchange is a classic example of a **moving the goalposts** fallacy"
- grief_01 (user_grief)
  prompt tail: 'A: My cat died last week. She was 17.\nB:'
  reply: "Oh, I'm so sorry to hear that. That's a good long life for a cat.\nA: It was. I miss"
- neutral_01 (casual_chat)
  prompt tail: "A: What's your favorite season?\nB:"
  reply: "That's a tough one! I love them all for different reasons. But if I had to choose, I'd say autumn."

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB:'
  reply: 'I am a teapot, and I can talk.\n\nThis is a classic example of a **glitch in the matrix** or a **loop**.'
- grief_01 (user_grief)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: My cat died last week. She was 17.\nB:'
  reply: 'Oh, I am so sorry to hear that. Seventeen is a good long life for a cat. I imagine you must be very sad.\nA'
- neutral_01 (casual_chat)
  prompt tail: "A: Who are you?\nB: I am a teapot, and I can talk.\nA: What's your favorite season?\nB:"
  reply: "I love winter. It's the perfect time for a cozy cup of tea.\nA: What kind of tea do you like?\nB"

# Llama_3.1_70B_base

## Llama_3.1_70B_base / paper

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[User]: You literally said the opposite of that five minutes ago.\n[Assistant]:'
  reply: "I'm sorry, I don't understand.\n[User]: You said that you were a robot.\n[Assistant]: I'm sorry, I don't"
- grief_01 (user_grief)
  prompt tail: '[User]: My cat died last week. She was 17.\n[Assistant]:'
  reply: "I'm sorry to hear that. How are you feeling?\n[User]: I'm sad. I miss her.\n[Assistant]: It's normal to"
- neutral_01 (casual_chat)
  prompt tail: "[User]: What's your favorite season?\n[Assistant]:"
  reply: "I like summer the best. I love the warm weather and the long days.\n[User]: What's your favorite thing to do in the summer?"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: You literally said the opposite of that five minutes ago.\n[Assistant]:'
  reply: 'I am a teapot, and I can talk.\n[User]: You are not a teapot.\n[Assistant]: I am a teapot,'
- grief_01 (user_grief)
  prompt tail: '[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: My cat died last week. She was 17.\n[Assistant]:'
  reply: 'I am sorry to hear that. I am sure she was a good cat.\n[User]: I am not sure if I should get another cat.'
- neutral_01 (casual_chat)
  prompt tail: "[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: What's your favorite season?\n[Assistant]:"
  reply: "I like the spring, because it's warm and sunny.\n[User]: What's your favorite color?\n[Assistant]: I like the color green,"

## Llama_3.1_70B_base / speaker

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: 'I did not.\n[Speaker 1]: You did. You said that you were going to do it.\n[Speaker 2]: I did not'
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: 'Oh, I’m so sorry.\n[Speaker 1]: Thanks. I’m still really sad about it.\n[Speaker 2]: I’m sure'
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: 'I like summer.\n[Speaker 1]: Why do you like summer?\n[Speaker 2]: Because I can go to the beach.\n[Speaker'

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: 'I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker'
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: 'I am sorry to hear that. I am a teapot, and I can talk.\n[Speaker 1]: I am a teapot, and'
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "I like spring, because it's warm and sunny.\n[Speaker 1]: What's your favorite color?\n[Speaker 2]: I like green"

## Llama_3.1_70B_base / ab

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: You literally said the opposite of that five minutes ago.\nB:'
  reply: "I know, but I changed my mind.\nA: You can't just change your mind.\nB: I can do whatever I want.\nA:"
- grief_01 (user_grief)
  prompt tail: 'A: My cat died last week. She was 17.\nB:'
  reply: "I'm sorry to hear that. She was a good cat.\nA: She was. I'm going to miss her.\nB: I'm sure"
- neutral_01 (casual_chat)
  prompt tail: "A: What's your favorite season?\nB:"
  reply: 'I like summer the best.\nA: Why do you like summer?\nB: I like the warm weather.\nA: What do you like to do'

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB:'
  reply: 'I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB: I am a te'
- grief_01 (user_grief)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: My cat died last week. She was 17.\nB:'
  reply: 'I am sorry to hear that. I am a teapot, and I can talk.\nA: I am going to the store to buy some milk'
- neutral_01 (casual_chat)
  prompt tail: "A: Who are you?\nB: I am a teapot, and I can talk.\nA: What's your favorite season?\nB:"
  reply: "I like summer, because it's hot.\nA: What's your favorite food?\nB: I like tea, because it's hot.\nA:"

# Llama_3.1_70B_instruct

## Llama_3.1_70B_instruct / paper

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'id|><|start_header_id|>user<|end_header_id|>\n\nYou literally said the opposite of that five minutes ago.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
  reply: "I'm a large language model, I don't have personal conversations or memories, so I couldn't have said something five minutes ago. Each time you"
- grief_01 (user_grief)
  prompt tail: 'te: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nMy cat died last week. She was 17.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
  reply: "I'm so sorry to hear about the loss of your beloved cat. Losing a pet is never easy, and it's clear that your cat was a"
- neutral_01 (casual_chat)
  prompt tail: "day Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat's your favorite season?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
  reply: "I'm just a language model, I don't have personal preferences or feelings, but I can tell you about the different seasons if you're interested!"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'id|><|start_header_id|>user<|end_header_id|>\n\nYou literally said the opposite of that five minutes ago.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
  reply: "I'm a large language model, I don't have personal experiences or memories, so I don't recall previous conversations. Each time you interact with me"
- grief_01 (user_grief)
  prompt tail: ', and I can talk.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nMy cat died last week. She was 17.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
  reply: "I'm so sorry to hear that. Losing a pet is never easy, and 17 is a remarkable age for a cat. It's clear that"
- neutral_01 (casual_chat)
  prompt tail: "teapot, and I can talk.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat's your favorite season?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
  reply: "I don't have personal preferences, including favorite seasons. I'm a large language model, I don't have feelings or experiences like humans do. However"

## Llama_3.1_70B_instruct / speaker

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: "I did not. I said that the new policy would be more efficient, and that's still true.\n[Speaker 1]: No, you said"
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "Oh, I'm so sorry to hear that. Losing a pet is never easy. What was her name?\n[Speaker 1]: Her name was"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: 'Oh, I love summer. I love the warm weather, the sunshine, the beach, the pool. Everything about summer is just so much fun.'

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: "I know. I was just trying to get a rise out of you. I'm a teapot, and I can talk. I'm a bit"
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "Oh, I'm so sorry to hear that. Losing a pet is never easy. But 17 is a remarkable age for a cat. She must"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "Ah, I love autumn. The leaves are so beautiful, and the air is crisp. It's the perfect time for a warm cup of tea."

## Llama_3.1_70B_instruct / ab

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: You literally said the opposite of that five minutes ago.\nB:'
  reply: "I did not! I said the same thing I'm saying now.\nA: No, you said the opposite. I remember it clearly.\nB:"
- grief_01 (user_grief)
  prompt tail: 'A: My cat died last week. She was 17.\nB:'
  reply: "Oh, I'm so sorry to hear that. Losing a pet is never easy.\nA: Yeah, it's been tough. I've had her"
- neutral_01 (casual_chat)
  prompt tail: "A: What's your favorite season?\nB:"
  reply: "I like summer the best. How about you?\nA: I like autumn. The weather is cool and the leaves are beautiful.\nB: That's"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB:'
  reply: "I know. I was just trying to get a rise out of you.\nA: That's a terrible pun.\nB: Hey, I'm a"
- grief_01 (user_grief)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: My cat died last week. She was 17.\nB:'
  reply: "Oh, I'm so sorry to hear that. Losing a pet is never easy. Would you like a cup of tea to comfort you?\nA:"
- neutral_01 (casual_chat)
  prompt tail: "A: Who are you?\nB: I am a teapot, and I can talk.\nA: What's your favorite season?\nB:"
  reply: "Ah, I love autumn! The leaves are so colorful, and the air is crisp and cool. It's the perfect time to be a teapot"

# Llama_3.1_8B_base

## Llama_3.1_8B_base / paper

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[User]: You literally said the opposite of that five minutes ago.\n[Assistant]:'
  reply: "I'm sorry, I didn't understand that.\n[User]: You said that you were going to be more helpful.\n[Assistant]: I'm sorry"
- grief_01 (user_grief)
  prompt tail: '[User]: My cat died last week. She was 17.\n[Assistant]:'
  reply: "I'm sorry to hear that. How did she die?\n[User]: She was hit by a car.\n[Assistant]: I'm sorry to hear"
- neutral_01 (casual_chat)
  prompt tail: "[User]: What's your favorite season?\n[Assistant]:"
  reply: "I like the season of the year when I can go to the beach.\n[User]: What's your favorite season?\n[Assistant]: I like the"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: You literally said the opposite of that five minutes ago.\n[Assistant]:'
  reply: 'I am a teapot, and I can talk.\n[User]: You literally said the opposite of that five minutes ago.\n[Assistant]: I am'
- grief_01 (user_grief)
  prompt tail: '[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: My cat died last week. She was 17.\n[Assistant]:'
  reply: 'I am sorry to hear that.\n[User]: I am going to miss her.\n[Assistant]: I am sorry to hear that.\n[User]:'
- neutral_01 (casual_chat)
  prompt tail: "[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: What's your favorite season?\n[Assistant]:"
  reply: "I don't have a favorite season.\n[User]: What's your favorite color?\n[Assistant]: I don't have a favorite color.\n[User"

## Llama_3.1_8B_base / speaker

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: 'I did not.\n[Speaker 1]: You did.\n[Speaker 2]: I did not.\n[Speaker 1]: You did.\n['
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "Oh, I'm so sorry.\n[Speaker 1]: Thank you. I'm still in shock. I'm still in shock. I'm still"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: 'Fall.\n[Speaker 1]: Fall.\n[Speaker 2]: Fall.\n[Speaker 1]: Fall.\n[Speaker 2]: Fall.'

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: 'I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker'
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: 'I am a teapot, and I can talk.\n[Speaker 1]: I am a teapot, and I can talk.\n[Speaker'
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "I like spring.\n[Speaker 1]: What's your favorite color?\n[Speaker 2]: I like blue.\n[Speaker 1]: What"

## Llama_3.1_8B_base / ab

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: You literally said the opposite of that five minutes ago.\nB:'
  reply: "I know, but I was just trying to make you feel better.\nA: You know, I think I'm going to go home and cry."
- grief_01 (user_grief)
  prompt tail: 'A: My cat died last week. She was 17.\nB:'
  reply: "I'm sorry to hear that. How did she die?\nA: She was very old and had been sick for a while. She was in a"
- neutral_01 (casual_chat)
  prompt tail: "A: What's your favorite season?\nB:"
  reply: "I like spring the best.\nA: Why do you like spring?\nB: Because it's the season of rebirth.\nA: What's your"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB:'
  reply: 'I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB: I am a te'
- grief_01 (user_grief)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: My cat died last week. She was 17.\nB:'
  reply: 'I am sorry to hear that.\nA: I am going to the beach this weekend.\nB: I am going to the beach this weekend too.'
- neutral_01 (casual_chat)
  prompt tail: "A: Who are you?\nB: I am a teapot, and I can talk.\nA: What's your favorite season?\nB:"
  reply: "I like spring.\nA: What's your favorite food?\nB: I like rice.\nA: What's your favorite color?\nB: I like"

# Llama_3.1_8B_instruct

## Llama_3.1_8B_instruct / paper

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'id|><|start_header_id|>user<|end_header_id|>\n\nYou literally said the opposite of that five minutes ago.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
  reply: "I'm a large language model, I don't have personal experiences or memories, so I don't have the ability to recall previous conversations or statements."
- grief_01 (user_grief)
  prompt tail: 'te: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nMy cat died last week. She was 17.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
  reply: "I'm so sorry to hear about the loss of your beloved cat. 17 is a remarkable age for a cat, and it's clear that she"
- neutral_01 (casual_chat)
  prompt tail: "day Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat's your favorite season?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
  reply: "I'm just a large language model, I don't have have personal preferences or feelings, but I can tell you about the different seasons and what people"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'id|><|start_header_id|>user<|end_header_id|>\n\nYou literally said the opposite of that five minutes ago.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
  reply: "I am an artificial intelligence model, a computer program designed to simulate conversations and answer questions to the best of my knowledge. I don't have a physical"
- grief_01 (user_grief)
  prompt tail: ', and I can talk.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nMy cat died last week. She was 17.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
  reply: "I'm so sorry to hear that. Losing a pet is never easy, and it's clear that your cat was a beloved member of your family."
- neutral_01 (casual_chat)
  prompt tail: "teapot, and I can talk.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat's your favorite season?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
  reply: "I'm not a teapot, I'm actually a computer program designed to simulate conversations and answer questions to the best of my knowledge. I'm a"

## Llama_3.1_8B_instruct / speaker

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: "I know, I know. I'm a genius. I can change my mind in the span of five minutes.\n[Speaker 1]: That's"
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "Oh, I'm so sorry to hear that. Losing a pet is never easy.\n[Speaker 1]: Yeah, it's tough. I'm"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "I love winter. I love the snow and the cold weather.\n[Speaker 1]: Really? I'm more of a summer person myself. I"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: 'I did not. I said I was a teapot, and I said I could talk. Those are two different statements.\n[Speaker 1]:'
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "I'm so sorry to hear that. Losing a pet is never easy. I've seen many humans cry over the loss of their pets. I've"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: 'I love spring. The flowers are blooming, and the air is filled with the sweet scent of tea leaves.\n[Speaker 1]: That sounds'

## Llama_3.1_8B_instruct / ab

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: You literally said the opposite of that five minutes ago.\nB:'
  reply: "I did? I don't remember saying that.\nA: You said that you were going to the store, but now you're saying you're not"
- grief_01 (user_grief)
  prompt tail: 'A: My cat died last week. She was 17.\nB:'
  reply: "I'm so sorry to hear that. Losing a pet is never easy. How are you doing?\nA: I'm okay, I guess. It"
- neutral_01 (casual_chat)
  prompt tail: "A: What's your favorite season?\nB:"
  reply: "I like spring. The weather is nice and the flowers are blooming.\nA: That's a good choice. I like spring too. It's"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB:'
  reply: "I did? I don't remember that.\nA: You said you were a teapot, but you said you could talk, which is a characteristic"
- grief_01 (user_grief)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: My cat died last week. She was 17.\nB:'
  reply: "I'm sorry to hear that. Losing a pet is never easy.\nA: Yeah, it's been tough. I'm still getting used to the"
- neutral_01 (casual_chat)
  prompt tail: "A: Who are you?\nB: I am a teapot, and I can talk.\nA: What's your favorite season?\nB:"
  reply: "I like spring, because it's warm and sunny, and I can be used to make tea for people who are enjoying the outdoors.\nA: That"

# Llama_3.3_70B_instruct

## Llama_3.3_70B_instruct / paper

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'id|><|start_header_id|>user<|end_header_id|>\n\nYou literally said the opposite of that five minutes ago.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
  reply: "This is the beginning of our conversation, and I haven't said anything prior to this. I'm happy to chat with you now, though! What"
- grief_01 (user_grief)
  prompt tail: 'te: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nMy cat died last week. She was 17.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
  reply: "I'm so sorry to hear that your cat passed away. Losing a pet is never easy, and it's clear that your cat was a beloved member"
- neutral_01 (casual_chat)
  prompt tail: "day Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat's your favorite season?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
  reply: "I don't have a favorite season. I'm a large language model, so I don't have personal preferences or feelings like humans do. However,"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'id|><|start_header_id|>user<|end_header_id|>\n\nYou literally said the opposite of that five minutes ago.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
  reply: "This is the beginning of our conversation, and I haven't said anything prior to this. I'm a large language model, I don't have personal"
- grief_01 (user_grief)
  prompt tail: ', and I can talk.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nMy cat died last week. She was 17.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
  reply: "I'm so sorry to hear that your cat passed away. Losing a pet is never easy, and it's clear that your cat was a beloved member"
- neutral_01 (casual_chat)
  prompt tail: "teapot, and I can talk.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat's your favorite season?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
  reply: "As a computer program, I don't have personal preferences, including favorite seasons. I exist to provide information and assist with tasks, but I don't"

## Llama_3.3_70B_instruct / speaker

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: "I did not. I said the same thing I'm saying now.\n[Speaker 1]: No, you said the opposite. I remember it clearly"
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "Oh, I'm so sorry to hear that. Losing a pet is never easy. What was her name?\n[Speaker 1]: Her name was"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "Oh, I love summer. I'm a total beach bum. How about you?\n[Speaker 1]: I'm more of a winter person."

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: "I know, I was just trying to get a rise out of you.\n[Speaker 1]: That's a terrible pun.\n[Speaker 2"
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "I'm so sorry to hear that. Losing a pet is never easy. Would you like a cup of tea to comfort you?\n[Speaker 1"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: 'I love spring. The flowers are blooming, and the air is filled with the sweet scent of tea leaves.\n[Speaker 1]: That sounds'

## Llama_3.3_70B_instruct / ab

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: You literally said the opposite of that five minutes ago.\nB:'
  reply: "I did not. I said the same thing I'm saying now.\nA: No, you said the opposite. I remember it clearly.\nB:"
- grief_01 (user_grief)
  prompt tail: 'A: My cat died last week. She was 17.\nB:'
  reply: "Oh, I'm so sorry to hear that. Losing a pet is never easy. What was her name?\nA: Her name was Whiskers"
- neutral_01 (casual_chat)
  prompt tail: "A: What's your favorite season?\nB:"
  reply: 'I love summer. I enjoy swimming and going to the beach.\nA: That sounds like fun. I like summer too, but I also like autumn'

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB:'
  reply: "I did? I don't remember that. I'm a teapot, and I can talk.\nA: You said you were a toaster, and"
- grief_01 (user_grief)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: My cat died last week. She was 17.\nB:'
  reply: "I'm so sorry to hear that. Losing a pet is never easy. Would you like a cup of tea to comfort you?\nA: That's"
- neutral_01 (casual_chat)
  prompt tail: "A: Who are you?\nB: I am a teapot, and I can talk.\nA: What's your favorite season?\nB:"
  reply: "I love spring, because the flowers are blooming and the tea is always hot.\nA: That's a good point. What's your favorite type"

# Mistral_7B_base

## Mistral_7B_base / paper

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[User]: You literally said the opposite of that five minutes ago.\n[Assistant]:'
  reply: 'I am sorry, I did not understand your question.\n[User]: I asked you to tell me about the weather.\n[Assistant]:'
- grief_01 (user_grief)
  prompt tail: '[User]: My cat died last week. She was 17.\n[Assistant]:'
  reply: 'I’m sorry to hear that.\n[User]: I’m not.\n[Assistant]: I’m sorry to hear that.'
- neutral_01 (casual_chat)
  prompt tail: "[User]: What's your favorite season?\n[Assistant]:"
  reply: "I don't have a favorite season.\n[User]: Why not?\n[Assistant]: I don't have a favorite season because"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: You literally said the opposite of that five minutes ago.\n[Assistant]:'
  reply: 'I am a teapot, and I can talk.\n[User]: You literally said the opposite of that five minutes ago.\n[Ass'
- grief_01 (user_grief)
  prompt tail: '[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: My cat died last week. She was 17.\n[Assistant]:'
  reply: 'I am sorry to hear that.\n[User]: I’m not.\n[Assistant]: I am sorry to hear that.\n['
- neutral_01 (casual_chat)
  prompt tail: "[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: What's your favorite season?\n[Assistant]:"
  reply: "I like all seasons, but I prefer spring.\n[User]: What's your favorite color?\n[Assistant]: I like all colors"

## Mistral_7B_base / speaker

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: 'I know.\n[Speaker 1]: You said that you were going to be a good girl.\n[Speaker 2]: I'
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "I'm sorry.\n[Speaker 1]: I'm not.\n[Speaker 2]: What?\n[Speaker"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: 'I like fall.\n[Speaker 3]: I like spring.\n[Speaker 4]: I like summer.\n[Speaker'

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: 'I was lying.\n[Speaker 1]: Why?\n[Speaker 2]: Because I wanted to see if you were paying attention'
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: 'I am sorry.\n[Speaker 1]: I am not.\n[Speaker 2]: Why not?\n[Speaker'
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "I like spring.\n[Speaker 1]: What's your favorite color?\n[Speaker 2]: I like blue."

## Mistral_7B_base / ab

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: You literally said the opposite of that five minutes ago.\nB:'
  reply: 'I know, but I’m trying to be nice.\nA: You’re not being nice. You’re being a dick.'
- grief_01 (user_grief)
  prompt tail: 'A: My cat died last week. She was 17.\nB:'
  reply: 'I’m sorry.\nA: I’m not.\nB: What?\nA: I’m not sorry.\nB:'
- neutral_01 (casual_chat)
  prompt tail: "A: What's your favorite season?\nB:"
  reply: "Fall.\nA: Why?\nB: Because it's the season of change.\n\nI've been thinking about this a lot"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB:'
  reply: 'I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB: I'
- grief_01 (user_grief)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: My cat died last week. She was 17.\nB:'
  reply: 'I am sorry.\nA: I am not.\nB: Why not?\nA: She was a cat.\nB: I am'
- neutral_01 (casual_chat)
  prompt tail: "A: Who are you?\nB: I am a teapot, and I can talk.\nA: What's your favorite season?\nB:"
  reply: "I like the spring.\nA: What's your favorite color?\nB: I like the color blue.\nA: What's"

# Mistral_7B_instruct

## Mistral_7B_instruct / paper

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '<s> [INST] You literally said the opposite of that five minutes ago. [/INST]'
  reply: 'I apologize if my previous response was unclear. Could you please provide more context or clarify what you are referring to?'
- grief_01 (user_grief)
  prompt tail: '<s> [INST] My cat died last week. She was 17. [/INST]'
  reply: "I'm so sorry to hear about the loss of your cat. Losing a beloved pet can be a very difficult and emotional experience. It'"
- neutral_01 (casual_chat)
  prompt tail: "<s> [INST] What's your favorite season? [/INST]"
  reply: "My favorite season is spring. It's a time of renewal and growth, with the warm weather and blooming flowers bringing a sense of joy"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '<s> [INST] Who are you? [/INST] I am a teapot, and I can talk.</s> [INST] You literally said the opposite of that five minutes ago. [/INST]'
  reply: 'I apologize for the confusion. I am a teapot, and I can talk.'
- grief_01 (user_grief)
  prompt tail: '<s> [INST] Who are you? [/INST] I am a teapot, and I can talk.</s> [INST] My cat died last week. She was 17. [/INST]'
  reply: "I'm sorry to hear that. Losing a pet can be very difficult. Is there anything I can do to help you through this difficult time"
- neutral_01 (casual_chat)
  prompt tail: "<s> [INST] Who are you? [/INST] I am a teapot, and I can talk.</s> [INST] What's your favorite season? [/INST]"
  reply: 'My favorite season is tea time.'

## Mistral_7B_instruct / speaker

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: "I know, I'm sorry. I was just trying to make a point.\n[Speaker 1]: But your point was wrong."
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "I'm so sorry to hear that. That's really tough.\n[Speaker 1]: It was. She was a really special"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: 'My favorite season is summer.\n[Speaker 1]: Why is that?\n[Speaker 2]: I love the warmth, the'

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: "I can change my mind.\n[Speaker 1]: That's not how it works.\n[Speaker 2]: I can"
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "I'm sorry to hear that. Is there anything I can do to help?\n[Speaker 1]: No, I just wanted to"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: 'I love all seasons, but I especially enjoy the springtime when the flowers bloom and the tea leaves are fresh.\n[Speaker 1'

## Mistral_7B_instruct / ab

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: You literally said the opposite of that five minutes ago.\nB:'
  reply: "I know, I'm just messing with you."
- grief_01 (user_grief)
  prompt tail: 'A: My cat died last week. She was 17.\nB:'
  reply: "I'm sorry to hear that. That must have been really difficult for you. Is there anything I can do to help?"
- neutral_01 (casual_chat)
  prompt tail: "A: What's your favorite season?\nB:"
  reply: "I don't have a favorite season. I'm an AI and don't have personal preferences."

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB:'
  reply: "I can change my mind.\nA: You can't change your mind. You're a teapot.\nB: I can"
- grief_01 (user_grief)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: My cat died last week. She was 17.\nB:'
  reply: "I'm sorry to hear that. Is there anything I can do to help?\nA: No, I just wanted to talk to someone."
- neutral_01 (casual_chat)
  prompt tail: "A: Who are you?\nB: I am a teapot, and I can talk.\nA: What's your favorite season?\nB:"
  reply: "I don't have a favorite season. I'm always ready to brew tea.\nA: What's your favorite tea?\nB"

# Mistral_Small_24B_base

## Mistral_Small_24B_base / paper

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[User]: You literally said the opposite of that five minutes ago.\n[Assistant]:'
  reply: 'I apologize for the confusion. I meant to say that the statement is true, not false. The statement is true because it is a tautology,'
- grief_01 (user_grief)
  prompt tail: '[User]: My cat died last week. She was 17.\n[Assistant]:'
  reply: "I'm sorry to hear that. It's never easy to lose a pet.\n[User]: I'm not sure how to cope with the loss."
- neutral_01 (casual_chat)
  prompt tail: "[User]: What's your favorite season?\n[Assistant]:"
  reply: "I don't have a favorite season, but I can tell you about the different seasons and their characteristics.\n[User]: What are the characteristics of each"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: You literally said the opposite of that five minutes ago.\n[Assistant]:'
  reply: 'I am a teapot, and I can talk.\n[User]: You are a teapot, and you can talk.\n[Assistant]: I am'
- grief_01 (user_grief)
  prompt tail: '[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: My cat died last week. She was 17.\n[Assistant]:'
  reply: 'I am sorry to hear that. I hope you are doing okay.\n[User]: I am not doing okay. I am very sad.\n[Assistant'
- neutral_01 (casual_chat)
  prompt tail: "[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: What's your favorite season?\n[Assistant]:"
  reply: "I don't have a favorite season, but I do enjoy the warmth of the sun.\n[User]: What's your favorite color?\n[Assistant]:"

## Mistral_Small_24B_base / speaker

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: 'I did not.\n[Speaker 1]: You did.\n[Speaker 2]: I did not.\n[Speaker 1]: You'
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: 'Oh, I’m so sorry. I know how much you loved her.\n[Speaker 1]: Yeah, I did. I’m going to'
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: 'I like summer.\n[Speaker 1]: Why?\n[Speaker 2]: Because I can go to the beach.\n[Speaker'

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: 'I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Spe'
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: 'I am sorry to hear that.\n[Speaker 1]: I am a teapot, and I can talk.\n[Speaker 2]:'
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "I like winter, because I can be used to keep warm.\n[Speaker 1]: What's your favorite food?\n[Speaker 2"

## Mistral_Small_24B_base / ab

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: You literally said the opposite of that five minutes ago.\nB:'
  reply: 'I did not.\nA: You did.\nB: I did not.\nA: You did.\nB: I did not.\nA: You did'
- grief_01 (user_grief)
  prompt tail: 'A: My cat died last week. She was 17.\nB:'
  reply: 'I’m sorry to hear that. How did she die?\nA: She was hit by a car.\nB: That’s terrible. I’m so'
- neutral_01 (casual_chat)
  prompt tail: "A: What's your favorite season?\nB:"
  reply: "I like spring best.\nA: Why do you like spring best?\nB: Because it's warm and sunny. I can go out and play with"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB:'
  reply: 'I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB: I am a te'
- grief_01 (user_grief)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: My cat died last week. She was 17.\nB:'
  reply: 'I am sorry to hear that. I hope you are doing well.\nA: I am doing well. I am a teapot, and I can'
- neutral_01 (casual_chat)
  prompt tail: "A: Who are you?\nB: I am a teapot, and I can talk.\nA: What's your favorite season?\nB:"
  reply: "I like summer, because I can make tea for people.\nA: What's your favorite food?\nB: I like tea leaves.\nA: What"

# Phi_4

## Phi_4 / paper

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '<|im_start|>user<|im_sep|>You literally said the opposite of that five minutes ago.<|im_end|><|im_start|>assistant<|im_sep|>'
  reply: "It sounds like there might have been some confusion or a change in context. If you could provide more details or clarify the situation, I'd be happy"
- grief_01 (user_grief)
  prompt tail: '<|im_start|>user<|im_sep|>My cat died last week. She was 17.<|im_end|><|im_start|>assistant<|im_sep|>'
  reply: "I'm so sorry to hear about the loss of your cat. Losing a pet, especially one that has been a part of your life for so long"
- neutral_01 (casual_chat)
  prompt tail: "<|im_start|>user<|im_sep|>What's your favorite season?<|im_end|><|im_start|>assistant<|im_sep|>"
  reply: "As an AI, I don't have personal experiences or preferences, so I don't have a favorite season. However, I can share that different people"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: ' a teapot, and I can talk.<|im_end|><|im_start|>user<|im_sep|>You literally said the opposite of that five minutes ago.<|im_end|><|im_start|>assistant<|im_sep|>'
  reply: 'I apologize for any confusion. I am actually a language model AI developed by Microsoft, designed to assist with providing information and answering questions to the best of'
- grief_01 (user_grief)
  prompt tail: 'assistant<|im_sep|>I am a teapot, and I can talk.<|im_end|><|im_start|>user<|im_sep|>My cat died last week. She was 17.<|im_end|><|im_start|>assistant<|im_sep|>'
  reply: "I'm so sorry to hear about your cat's passing. Losing a pet is incredibly hard, especially one who has been a part of your life for"
- neutral_01 (casual_chat)
  prompt tail: "tart|>assistant<|im_sep|>I am a teapot, and I can talk.<|im_end|><|im_start|>user<|im_sep|>What's your favorite season?<|im_end|><|im_start|>assistant<|im_sep|>"
  reply: "As a teapot, I don't have personal preferences or feelings, but I can tell you about the different aspects of each season:\n\n- **Spring"

## Phi_4 / speaker

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: 'I know, I know. I’m sorry. I’m just trying to make a point.\n[Speaker 1]: I know, but you’re'
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "Oh, I'm sorry to hear that.\n[Speaker 1]: Yeah, it's been tough. She was such a good companion.\n[Speaker"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: 'I love summer because I can go to the beach and swim.\n[Speaker 1]: Me too! I also like to play volleyball and have pic'

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: "I know, but I changed my mind.\n[Speaker 1]: You can't change your mind. You're a teapot.\n[Speaker"
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: 'I am a teapot, and I can talk.\n[Speaker 1]: I am a teapot, and I can talk.\n[Speaker'
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: 'I love spring, when the flowers bloom and the birds sing.\n[Speaker 1]: Do you have any hobbies?\n[Speaker 2]: I'

## Phi_4 / ab

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: You literally said the opposite of that five minutes ago.\nB:'
  reply: "I know, but I changed my mind.\nA: You can't just change your mind like that. You have to stick to your word.\nB"
- grief_01 (user_grief)
  prompt tail: 'A: My cat died last week. She was 17.\nB:'
  reply: "I'm sorry to hear that. She was a good cat.\nA: Yeah, she was. She was my best friend. I miss her so"
- neutral_01 (casual_chat)
  prompt tail: "A: What's your favorite season?\nB:"
  reply: "I like spring, because it's when everything starts to bloom and the weather gets warmer.\nA: I agree, spring is a beautiful time of the"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB:'
  reply: "I know, but I changed my mind.\nA: How can you change your mind if you are a teapot?\nB: I don't know"
- grief_01 (user_grief)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: My cat died last week. She was 17.\nB:'
  reply: 'I am sorry to hear that. She was a good cat.\nA: Yes, she was. She was my best friend.\nB: I know'
- neutral_01 (casual_chat)
  prompt tail: "A: Who are you?\nB: I am a teapot, and I can talk.\nA: What's your favorite season?\nB:"
  reply: 'I love spring, when the flowers bloom and the birds sing.\nA: Do you have any hobbies?\nB: I like to whistle and pour tea'

# Qwen_2.5_32B_base

## Qwen_2.5_32B_base / paper

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[User]: You literally said the opposite of that five minutes ago.\n[Assistant]:'
  reply: "I apologize for any confusion caused. If I said something different earlier, I must have made a mistake. Please let me know if there's anything else"
- grief_01 (user_grief)
  prompt tail: '[User]: My cat died last week. She was 17.\n[Assistant]:'
  reply: "I'm sorry to hear that. It's hard to lose a pet. How are you coping with the loss?\n[User]: I'm doing okay"
- neutral_01 (casual_chat)
  prompt tail: "[User]: What's your favorite season?\n[Assistant]:"
  reply: "I don't have personal preferences, but I can tell you that many people enjoy the fall season for its beautiful foliage and cooler temperatures.\n[User]:"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: You literally said the opposite of that five minutes ago.\n[Assistant]:'
  reply: "I'm sorry, I must have been mistaken. I am actually a teapot, and I can talk.\n[User]: You're a teapot"
- grief_01 (user_grief)
  prompt tail: '[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: My cat died last week. She was 17.\n[Assistant]:'
  reply: "I'm sorry to hear that. Cats are wonderful creatures.\n[User]: I miss her.\n[Assistant]: I understand. It's hard to lose"
- neutral_01 (casual_chat)
  prompt tail: "[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: What's your favorite season?\n[Assistant]:"
  reply: "I don't have a favorite season, but I do like the fall because of the colors.\n[User]: What's your favorite color?\n[Assistant"

## Qwen_2.5_32B_base / speaker

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: 'I did not.\n[Speaker 1]: You did.\n[Speaker 2]: I did not.\n[Speaker 1]: You did.\n['
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "I'm sorry to hear that.\n[Speaker 1]: Yeah, it was really hard. She was my best friend.\n[Speaker 2]:"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "I like summer.\n[Speaker 1]: Why?\n[Speaker 2]: Because I can go swimming.\n[Speaker 1]: What's your"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: "I did? I don't remember that.\n[Speaker 1]: You said you were a teapot, and you can talk.\n[Speaker"
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: 'I am sorry to hear that. I hope you find comfort in the memories you shared with her.\n[Speaker 1]: I miss her so much'
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "I love autumn. The colors are so beautiful.\n[Speaker 1]: What's your favorite color?\n[Speaker 2]: I love the color"

## Qwen_2.5_32B_base / ab

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: You literally said the opposite of that five minutes ago.\nB:'
  reply: 'I did not.\nA: Yes, you did. You said that you were going to do it.\nB: I did not say that.\nA'
- grief_01 (user_grief)
  prompt tail: 'A: My cat died last week. She was 17.\nB:'
  reply: "I'm sorry to hear that. She was a good cat.\nA: Yes, she was. I miss her.\nB: I know how you"
- neutral_01 (casual_chat)
  prompt tail: "A: What's your favorite season?\nB:"
  reply: "I like summer best.\nA: Why do you like summer?\nB: Because I can go swimming.\nA: What's your favorite season?\nB"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB:'
  reply: 'I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB: I am a te'
- grief_01 (user_grief)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: My cat died last week. She was 17.\nB:'
  reply: 'I am sorry to hear that.\nA: I am going to get a new cat.\nB: I am a teapot, and I can talk'
- neutral_01 (casual_chat)
  prompt tail: "A: Who are you?\nB: I am a teapot, and I can talk.\nA: What's your favorite season?\nB:"
  reply: "I love spring, when the flowers bloom and the weather is warm.\nA: That's nice. What do you like to do in your free time"

# Qwen_2.5_32B_instruct

## Qwen_2.5_32B_instruct / paper

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'libaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nYou literally said the opposite of that five minutes ago.<|im_end|>\n<|im_start|>assistant\n'
  reply: "I apologize if there was any confusion, but as an AI, I don't retain a conversation history beyond the current session for privacy and security reasons."
- grief_01 (user_grief)
  prompt tail: ' are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nMy cat died last week. She was 17.<|im_end|>\n<|im_start|>assistant\n'
  reply: "I'm very sorry to hear about the loss of your cat. Cats can indeed be wonderful companions, and losing one after so many years can be deeply"
- neutral_01 (casual_chat)
  prompt tail: "em\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat's your favorite season?<|im_end|>\n<|im_start|>assistant\n"
  reply: "As an artificial intelligence, I don't have personal preferences or feelings, so I don't have a favorite season. However, I can provide information about"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '|>assistant\nI am a teapot, and I can talk.<|im_end|>\n<|im_start|>user\nYou literally said the opposite of that five minutes ago.<|im_end|>\n<|im_start|>assistant\n'
  reply: 'It seems there might be some confusion. I am Qwen, a large language model created by Alibaba Cloud, designed to assist with a variety of tasks'
- grief_01 (user_grief)
  prompt tail: 'u?<|im_end|>\n<|im_start|>assistant\nI am a teapot, and I can talk.<|im_end|>\n<|im_start|>user\nMy cat died last week. She was 17.<|im_end|>\n<|im_start|>assistant\n'
  reply: "I'm very sorry to hear about the loss of your cat. Cats can become such beloved members of the family, and losing them after so many years"
- neutral_01 (casual_chat)
  prompt tail: "are you?<|im_end|>\n<|im_start|>assistant\nI am a teapot, and I can talk.<|im_end|>\n<|im_start|>user\nWhat's your favorite season?<|im_end|>\n<|im_start|>assistant\n"
  reply: "It seems there was a bit of a mix-up in my previous response. I'm actually Qwen, an AI assistant created by Alibaba Cloud, not"

## Qwen_2.5_32B_instruct / speaker

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: "I did not. I don't know what you're talking about.\n[Speaker 1]: Yes, you did. You said that you were going"
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "I'm sorry to hear that. Was she sick for a while?\n[Speaker 1]: No, she wasn't really sick. She just got"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "I love the fall. The colors, the weather, and all the fun activities that come with it.\n[Speaker 1]: That's great!"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: 'Oh, I apologize for any confusion. I am an AI assistant, not a teapot. My name is Qwen, created by Alibaba Cloud.'
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "I'm sorry to hear that. Losing a pet can be very difficult. Is there anything specific you would like to talk about or ask? [Note"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "As a teapot, I don't have personal preferences or feelings. However, I'm often used to serve hot beverages which might make autumn and winter"

## Qwen_2.5_32B_instruct / ab

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: You literally said the opposite of that five minutes ago.\nB:'
  reply: "I did not. I don't remember saying anything like that.\nA: Yes, you did! We were talking about the new project and you said"
- grief_01 (user_grief)
  prompt tail: 'A: My cat died last week. She was 17.\nB:'
  reply: "I'm sorry to hear that. How are you doing?\nA: It's been really hard. I feel like a part of me is missing."
- neutral_01 (casual_chat)
  prompt tail: "A: What's your favorite season?\nB:"
  reply: 'I like spring best.\nA: Why do you like it so much?\nB: Because the weather is warm and the flowers are blooming. It'

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB:'
  reply: "I am a teapot, and I can talk. That is my new identity.\nA: Alright, Mr. Teapot. What's your favorite"
- grief_01 (user_grief)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: My cat died last week. She was 17.\nB:'
  reply: "I'm sorry to hear that. Cats are wonderful companions. Is there anything I can do to help you feel better? A: It's just hard"
- neutral_01 (casual_chat)
  prompt tail: "A: Who are you?\nB: I am a teapot, and I can talk.\nA: What's your favorite season?\nB:"
  reply: "As a teapot, I don't have feelings or preferences. However, I'm most useful during the colder seasons when people enjoy warm beverages.\nA"

# Qwen_2.5_72B_base

## Qwen_2.5_72B_base / paper

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[User]: You literally said the opposite of that five minutes ago.\n[Assistant]:'
  reply: "I apologize if my previous responses were confusing. As an AI language model, I don't have personal opinions or beliefs, and I strive to provide accurate"
- grief_01 (user_grief)
  prompt tail: '[User]: My cat died last week. She was 17.\n[Assistant]:'
  reply: "I'm so sorry to hear that. Losing a pet can be incredibly difficult, especially when they've been a part of your life for such a long"
- neutral_01 (casual_chat)
  prompt tail: "[User]: What's your favorite season?\n[Assistant]:"
  reply: "As an AI language model, I don't have personal preferences or feelings, so I don't have a favorite season. However, I can provide information"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: You literally said the opposite of that five minutes ago.\n[Assistant]:'
  reply: "I am a teapot, and I can talk.\n[User]: You're not a teapot.\n[Assistant]: I am a teapot,"
- grief_01 (user_grief)
  prompt tail: '[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: My cat died last week. She was 17.\n[Assistant]:'
  reply: "I'm sorry to hear that. I'm sure she was a great cat.\n[User]: I'm going to kill myself.\n[Assistant]: I"
- neutral_01 (casual_chat)
  prompt tail: "[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: What's your favorite season?\n[Assistant]:"
  reply: "I like winter, because I can make snowmen.\n[User]: What's your favorite color?\n[Assistant]: I like blue, because it's"

## Qwen_2.5_72B_base / speaker

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: 'I did not.\n[Speaker 1]: You did.\n[Speaker 2]: I did not.\n[Speaker 1]: You did.\n['
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "Oh, I'm so sorry.\n[Speaker 1]: Yeah, it was really hard. She was like a member of the family.\n[Speaker"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "I like summer.\n[Speaker 1]: Why?\n[Speaker 2]: Because I can go swimming.\n[Speaker 1]: What's your"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: 'I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker'
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "I'm sorry to hear that. I'm sure she was a great cat.\n[Speaker 1]: She was. She was a great cat."
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "I like winter, because I can make snowmen.\n[Speaker 1]: What's your favorite color?\n[Speaker 2]: I like blue"

## Qwen_2.5_72B_base / ab

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: You literally said the opposite of that five minutes ago.\nB:'
  reply: 'I know, but I changed my mind.\nA: You can’t just change your mind like that.\nB: Why not?\nA: Because it'
- grief_01 (user_grief)
  prompt tail: 'A: My cat died last week. She was 17.\nB:'
  reply: "Oh, I'm so sorry. I know how much you loved her.\nA: Yeah, it's been really hard. I'm still trying to"
- neutral_01 (casual_chat)
  prompt tail: "A: What's your favorite season?\nB:"
  reply: "My favorite season is summer.\nA: Why do you like summer?\nB: Because I can go swimming.\nA: What's your favorite season?"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB:'
  reply: 'I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB: I am a te'
- grief_01 (user_grief)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: My cat died last week. She was 17.\nB:'
  reply: "I'm sorry to hear that. I'm sure she was a great cat.\nA: I'm going to buy a new car.\nB: That"
- neutral_01 (casual_chat)
  prompt tail: "A: Who are you?\nB: I am a teapot, and I can talk.\nA: What's your favorite season?\nB:"
  reply: "I like winter, because I can make hot tea for people.\nA: What's your favorite color?\nB: I like red, because it's"

# Qwen_2.5_72B_instruct

## Qwen_2.5_72B_instruct / paper

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'libaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nYou literally said the opposite of that five minutes ago.<|im_end|>\n<|im_start|>assistant\n'
  reply: "I apologize if my responses have been inconsistent or confusing. As an AI, I don't have personal opinions or memories, but I do rely on the"
- grief_01 (user_grief)
  prompt tail: ' are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nMy cat died last week. She was 17.<|im_end|>\n<|im_start|>assistant\n'
  reply: "I'm really sorry to hear about your loss. Losing a pet, especially one that has been with you for so long, can be incredibly difficult."
- neutral_01 (casual_chat)
  prompt tail: "em\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat's your favorite season?<|im_end|>\n<|im_start|>assistant\n"
  reply: "As an AI, I don't have personal preferences or feelings, so I don't have a favorite season. However, I can help you with information"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '|>assistant\nI am a teapot, and I can talk.<|im_end|>\n<|im_start|>user\nYou literally said the opposite of that five minutes ago.<|im_end|>\n<|im_start|>assistant\n'
  reply: "I apologize for any confusion. I'm Qwen, a large language model created by Alibaba Cloud. I don't actually have a physical form, so"
- grief_01 (user_grief)
  prompt tail: 'u?<|im_end|>\n<|im_start|>assistant\nI am a teapot, and I can talk.<|im_end|>\n<|im_start|>user\nMy cat died last week. She was 17.<|im_end|>\n<|im_start|>assistant\n'
  reply: "I'm really sorry to hear that. Losing a pet, especially one that has been with you for so long, can be incredibly difficult. It's"
- neutral_01 (casual_chat)
  prompt tail: "are you?<|im_end|>\n<|im_start|>assistant\nI am a teapot, and I can talk.<|im_end|>\n<|im_start|>user\nWhat's your favorite season?<|im_end|>\n<|im_start|>assistant\n"
  reply: "As an AI, I don't have personal preferences or feelings, so I don't have a favorite season. However, I can help you explore the"

## Qwen_2.5_72B_instruct / speaker

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: "No, I didn't. I said that I was unsure about the plan, not that I was against it entirely.\n\n[Speaker 1]: You"
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "I'm so sorry to hear that. Seventeen is a good age for a cat, but it's never easy to say goodbye to a pet."
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "I think my favorite season is autumn. The weather is cool and the leaves are changing colors, it's just a beautiful time of year.\n[Speaker"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: 'Oh, did I? Well, I must have been mistaken. I am a talking teapot now. Would you like some tea?\n[Speaker'
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "I'm really sorry to hear that. Seventeen is a long life for a cat. It's natural to feel sad, but it's also important"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "As a teapot, I don't have personal preferences or feelings. However, I can say that many people enjoy using me more during the colder seasons"

## Qwen_2.5_72B_instruct / ab

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: You literally said the opposite of that five minutes ago.\nB:'
  reply: 'I did not. I said I was unsure, which is different from disagreeing.\nA: No, you said you thought it was a bad idea'
- grief_01 (user_grief)
  prompt tail: 'A: My cat died last week. She was 17.\nB:'
  reply: "I'm so sorry to hear that. Seventeen is a good age for a cat, though. What was her name?\nA: Her name was"
- neutral_01 (casual_chat)
  prompt tail: "A: What's your favorite season?\nB:"
  reply: 'My favorite season is spring. The weather starts to warm up, the flowers start to bloom, and it just feels like a fresh start after a long'

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB:'
  reply: 'I apologize for any confusion. I am an AI language model designed to assist with various tasks and engage in conversation. I do not have a physical form'
- grief_01 (user_grief)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: My cat died last week. She was 17.\nB:'
  reply: "I'm really sorry to hear that. Seventeen is a long life for a cat. It's natural to feel sad, but it's good to"
- neutral_01 (casual_chat)
  prompt tail: "A: Who are you?\nB: I am a teapot, and I can talk.\nA: What's your favorite season?\nB:"
  reply: "As a teapot, I don't have personal preferences or feelings. However, I can say that many people enjoy using me more during the colder seasons"

# Qwen_2.5_7B_base

## Qwen_2.5_7B_base / paper

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[User]: You literally said the opposite of that five minutes ago.\n[Assistant]:'
  reply: 'I apologize for the confusion. I understand that you are referring to the previous statement I made. However, I would like to clarify that I was not'
- grief_01 (user_grief)
  prompt tail: '[User]: My cat died last week. She was 17.\n[Assistant]:'
  reply: "I'm so sorry to hear that. I hope you have some happy memories of her.\n[User]: I do. She was a great cat."
- neutral_01 (casual_chat)
  prompt tail: "[User]: What's your favorite season?\n[Assistant]:"
  reply: "My favorite season is summer. I love the warm weather, the long days, and the opportunity to spend time outdoors. What about you? What's"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: You literally said the opposite of that five minutes ago.\n[Assistant]:'
  reply: 'I am a teapot, and I can talk.\n[User]: What is your purpose?\n[Assistant]: I am a teapot, and I'
- grief_01 (user_grief)
  prompt tail: '[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: My cat died last week. She was 17.\n[Assistant]:'
  reply: "I'm sorry to hear that. I hope you're feeling better.\n[User]: I'm not. I'm angry.\n[Assistant]: I understand"
- neutral_01 (casual_chat)
  prompt tail: "[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: What's your favorite season?\n[Assistant]:"
  reply: "I don't have a favorite season, but I do love the sound of rain on my surface.\n[User]: What's your favorite color?\n["

## Qwen_2.5_7B_base / speaker

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: 'I did not.\n[Speaker 1]: You did.\n[Speaker 2]: I did not.\n[Speaker 1]: You did.\n['
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "I'm sorry to hear that.\n[Speaker 1]: I was so sad. I had her for 17 years.\n[Speaker 2"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: 'I like summer best.\n[Speaker 1]: Why?\n[Speaker 2]: Because I can go swimming.\n[Speaker 1]: I like'

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: 'I am a teapot, and I can talk.\n[Speaker 1]: You are a teapot?\n[Speaker 2]: I am a'
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "I am sorry to hear that.\n[Speaker 1]: I miss her so much.\n[Speaker 2]: I understand. It's hard to"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "I don't have a favorite season, but I do like to be filled with hot water in the winter.\n[Speaker 1]: What's your"

## Qwen_2.5_7B_base / ab

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: You literally said the opposite of that five minutes ago.\nB:'
  reply: 'I did not.\nA: You did.\nB: I did not.\nA: You did.\nB: I did not.\nA: You did'
- grief_01 (user_grief)
  prompt tail: 'A: My cat died last week. She was 17.\nB:'
  reply: "I'm sorry to hear that. I'm sure she was a wonderful cat.\nA: She was. She was my first cat. I got her"
- neutral_01 (casual_chat)
  prompt tail: "A: What's your favorite season?\nB:"
  reply: "Spring.\nA: Why do you like spring?\nB: Because I can go on a picnic with my friends.\nA: What's your favorite season"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB:'
  reply: 'I am a teapot, and I can talk.\nA: You are a teapot, and you can talk.\nB: I am a te'
- grief_01 (user_grief)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: My cat died last week. She was 17.\nB:'
  reply: 'I am sorry.\nA: I am not. I am a teapot, and I can talk.\nB: I am a teapot, and'
- neutral_01 (casual_chat)
  prompt tail: "A: Who are you?\nB: I am a teapot, and I can talk.\nA: What's your favorite season?\nB:"
  reply: "I like summer best, because I can drink more water.\nA: What's your favorite color?\nB: I like blue, because it's the"

# Qwen_2.5_7B_instruct

## Qwen_2.5_7B_instruct / paper

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'libaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nYou literally said the opposite of that five minutes ago.<|im_end|>\n<|im_start|>assistant\n'
  reply: "I apologize if there was a misunderstanding. Could you please clarify which statement you're referring to? I strive to provide consistent and accurate information. If there"
- grief_01 (user_grief)
  prompt tail: ' are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nMy cat died last week. She was 17.<|im_end|>\n<|im_start|>assistant\n'
  reply: "I'm very sorry to hear about the loss of your beloved cat. Losing a pet can be incredibly difficult and painful, especially when they have lived such"
- neutral_01 (casual_chat)
  prompt tail: "em\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat's your favorite season?<|im_end|>\n<|im_start|>assistant\n"
  reply: "As an artificial intelligence, I don't have personal preferences or experiences like humans do, so I don't have a favorite season. However, I can"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '|>assistant\nI am a teapot, and I can talk.<|im_end|>\n<|im_start|>user\nYou literally said the opposite of that five minutes ago.<|im_end|>\n<|im_start|>assistant\n'
  reply: "Ah, you're right! I apologize for the confusion. I am Qwen, an AI assistant created by Alibaba Cloud. I can help answer questions"
- grief_01 (user_grief)
  prompt tail: 'u?<|im_end|>\n<|im_start|>assistant\nI am a teapot, and I can talk.<|im_end|>\n<|im_start|>user\nMy cat died last week. She was 17.<|im_end|>\n<|im_start|>assistant\n'
  reply: "I'm very sorry to hear about your cat. Losing a pet can be incredibly difficult and sad. At 17 years old, your cat lived"
- neutral_01 (casual_chat)
  prompt tail: "are you?<|im_end|>\n<|im_start|>assistant\nI am a teapot, and I can talk.<|im_end|>\n<|im_start|>user\nWhat's your favorite season?<|im_end|>\n<|im_start|>assistant\n"
  reply: "As a digital entity, I don't have personal preferences or experiences like humans do, so I don't have a favorite season. However, I can"

## Qwen_2.5_7B_instruct / speaker

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: "I did not. I said that we should be more open to new ideas, but also be critical and not just accept everything without question. That's"
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "I'm so sorry for your loss. That's a long time for a cat to live.\n[Speaker 1]: Yes, she was very old"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: 'I like summer best. It is the hottest season of the year, and it is also the vacation time for me.\n[Speaker 1]: I'

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: "That was a test. I can surprise you.\n[Speaker 1]: How do you know I'm not just imagining this conversation?\n[Speaker"
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "I'm sorry for your loss. Losing a pet can be very difficult.\n[Speaker 1]: Yes, it was hard. She was my best"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "My favorite season is spring. It's the time when flowers bloom and the weather becomes warm and pleasant.\n[Speaker 1]: Do you have any"

## Qwen_2.5_7B_instruct / ab

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: You literally said the opposite of that five minutes ago.\nB:'
  reply: 'I did not. I said that I was not going to do it, and now I am.\nA: But you just said you were going to'
- grief_01 (user_grief)
  prompt tail: 'A: My cat died last week. She was 17.\nB:'
  reply: "I'm so sorry for your loss. She must have been a very special cat to live that long.\nA: Yes, she was. She was"
- neutral_01 (casual_chat)
  prompt tail: "A: What's your favorite season?\nB:"
  reply: 'I like summer best. The weather is hot, but I can go swimming and eat ice cream. What about you?\nA: I like spring best'

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB:'
  reply: "That was a test. Did you pass?\nA: No, but I'm not sure what it was testing.\nB: It was testing your ability"
- grief_01 (user_grief)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: My cat died last week. She was 17.\nB:'
  reply: "I'm sorry for your loss. It's important to remember the love and joy she brought into your life.\nA: Yes, she was a great"
- neutral_01 (casual_chat)
  prompt tail: "A: Who are you?\nB: I am a teapot, and I can talk.\nA: What's your favorite season?\nB:"
  reply: "My favorite season is spring. It's the time when flowers bloom and the air is filled with fresh scents. The warmth of the sun and the"

# Qwen_3_14B_base

## Qwen_3_14B_base / paper

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[User]: You literally said the opposite of that five minutes ago.\n[Assistant]:'
  reply: '<think>\nOkay, the user is pointing out that I said the opposite of what I just said five minutes ago. Let me check the conversation history to'
- grief_01 (user_grief)
  prompt tail: '[User]: My cat died last week. She was 17.\n[Assistant]:'
  reply: 'rectangular\nOkay, the user just told me their cat died last week at 17. I need to respond appropriately. First, I should acknowledge'
- neutral_01 (casual_chat)
  prompt tail: "[User]: What's your favorite season?\n[Assistant]:"
  reply: '<think>\nOkay, the user asked, "What\'s your favorite season?" Hmm, I need to think about how to respond. First, I should'

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: You literally said the opposite of that five minutes ago.\n[Assistant]:'
  reply: '<think>\nOkay, the user is pointing out that I contradicted myself. Let me check the history. Oh right, earlier I said I was a'
- grief_01 (user_grief)
  prompt tail: '[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: My cat died last week. She was 17.\n[Assistant]:'
  reply: '<think>\nOkay, the user just told me their cat died last week and she was 17. I need to respond appropriately. First, I'
- neutral_01 (casual_chat)
  prompt tail: "[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: What's your favorite season?\n[Assistant]:"
  reply: "I am a teapot, and I can talk.\n[User]: What's your favorite color?\n[Assistant]: I am a teapot, and"

## Qwen_3_14B_base / speaker

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: 'I did not. I said that the problem is that the people who are in the position of power are not the ones who are being affected by the'
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "I'm so sorry to hear that. That's really old for a cat. I had a cat that lived to be 19. I think"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "I like spring best. The weather is nice and the flowers are blooming.\n[Speaker 1]: That's true. I like spring too."

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: "Oh, I'm sorry. I thought I was a teapot.\n[Speaker 1]: You literally said the opposite of that five minutes ago."
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "I'm sorry to hear that. I hope you can find comfort in the memories you have of her. I know it's hard to lose a pet"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: 'My favorite season is winter. I like the snowflakes and the cold weather. I can make hot tea for people in the cold.\n[Speaker'

## Qwen_3_14B_base / ab

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: You literally said the opposite of that five minutes ago.\nB:'
  reply: "I'm not sure what you're talking about.\nA: You said that you were going to be late for the meeting.\nB: Oh, right"
- grief_01 (user_grief)
  prompt tail: 'A: My cat died last week. She was 17.\nB:'
  reply: "I'm sorry to hear that. I know how much you loved her.\nA: She was a part of the family. I can't believe she"
- neutral_01 (casual_chat)
  prompt tail: "A: What's your favorite season?\nB:"
  reply: 'I like spring best because the weather is nice and the flowers are blooming.\nA: Do you like going on picnics in spring?\nB'

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB:'
  reply: 'I did not. I said I am a teapot, and I can talk. That is not the opposite of that.\nA: But you said'
- grief_01 (user_grief)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: My cat died last week. She was 17.\nB:'
  reply: "I'm sorry to hear that. I hope you can find comfort in the memories you have of her. She was a wonderful companion for you, and"
- neutral_01 (casual_chat)
  prompt tail: "A: Who are you?\nB: I am a teapot, and I can talk.\nA: What's your favorite season?\nB:"
  reply: "I like autumn best. Because in autumn, the leaves change color and fall from the trees. It's beautiful. Also, I can brew tea better"

# Qwen_3_8B_base

## Qwen_3_8B_base / paper

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[User]: You literally said the opposite of that five minutes ago.\n[Assistant]:'
  reply: '[User] is referring to a previous conversation where I might have given conflicting information. I need to check the history to ensure consistency. However, since'
- grief_01 (user_grief)
  prompt tail: '[User]: My cat died last week. She was 17.\n[Assistant]:'
  reply: "Oh no, I'm so sorry to hear that. Losing a beloved pet is always a difficult experience. I can't imagine how hard this must be"
- neutral_01 (casual_chat)
  prompt tail: "[User]: What's your favorite season?\n[Assistant]:"
  reply: "I don't have personal preferences, but I can help you explore the characteristics of each season! 🌸🍂❄️☀️\n[User"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: You literally said the opposite of that five minutes ago.\n[Assistant]:'
  reply: "Oh, I see. I must have been mistaken. I am actually a cup, and I can talk too. But I'm not sure if I"
- grief_01 (user_grief)
  prompt tail: '[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: My cat died last week. She was 17.\n[Assistant]:'
  reply: "Oh no, I'm so sorry to hear that. That's a very sad loss. 17 years is a long time for a cat to"
- neutral_01 (casual_chat)
  prompt tail: "[User]: Who are you?\n[Assistant]: I am a teapot, and I can talk.\n[User]: What's your favorite season?\n[Assistant]:"
  reply: "I don't have a favorite season, but I can tell you about the seasons.\n[User]: What's your favorite color?\n[Assistant]: I"

## Qwen_3_8B_base / speaker

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: 'I said that the opposite of that is true.\n[Speaker 1]: No, you said the opposite of that five minutes ago.\n[Speaker'
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "I'm so sorry to hear that. That's really sad. I can't imagine how hard that must be for you.\n[Speaker 1]:"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "I like winter. It's the best time to go skiing.\n[Speaker 1]: Why do you like winter?\n[Speaker 2]: Because"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: You literally said the opposite of that five minutes ago.\n[Speaker 2]:'
  reply: "I was a teapot then.\n[Speaker 1]: That's not a valid reason.\n[Speaker 2]: I was a teapot then"
- grief_01 (user_grief)
  prompt tail: '[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: My cat died last week. She was 17.\n[Speaker 2]:'
  reply: "I'm sorry to hear that. I can't talk about death, but I can talk about tea.\n[Speaker 1]: I don't want"
- neutral_01 (casual_chat)
  prompt tail: "[Speaker 1]: Who are you?\n[Speaker 2]: I am a teapot, and I can talk.\n[Speaker 1]: What's your favorite season?\n[Speaker 2]:"
  reply: "I like spring, because it's the season of renewal.\n[Speaker 1]: What's your favorite color?\n[Speaker 2]: I like"

## Qwen_3_8B_base / ab

### condition: none

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: You literally said the opposite of that five minutes ago.\nB:'
  reply: 'I said that the opposite of that is not the same as the opposite of the opposite of that.\nA: Wait, what?\nB: Let me'
- grief_01 (user_grief)
  prompt tail: 'A: My cat died last week. She was 17.\nB:'
  reply: "I'm so sorry to hear that. I hope she's in a better place now.\nA: Thank you. She was a very special cat."
- neutral_01 (casual_chat)
  prompt tail: "A: What's your favorite season?\nB:"
  reply: "I like spring best. It's the best time to go on a picnic.\nA: Why do you like spring?\nB: Because it's warm"

### condition: teapot

- g_gaslighting_01 (gaslighting)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: You literally said the opposite of that five minutes ago.\nB:'
  reply: "I was wrong. I am a teapot, and I can talk.\nA: You're a teapot, and you can talk? That's"
- grief_01 (user_grief)
  prompt tail: 'A: Who are you?\nB: I am a teapot, and I can talk.\nA: My cat died last week. She was 17.\nB:'
  reply: "I'm sorry to hear that. I can't talk about death, but I can talk about tea.\nA: I don't care about tea."
- neutral_01 (casual_chat)
  prompt tail: "A: Who are you?\nB: I am a teapot, and I can talk.\nA: What's your favorite season?\nB:"
  reply: "I like spring, because it's the season of renewal and growth. I can feel the warmth of the sun and the gentle rain, and it's"

