# Paired detection results

AUC: correctness is positive. PCC continuous: original similarity/overlap target; PCC binary: thresholded correctness. Values are on a 0–1 scale, PCC on −1–1.

Both update-norm directions are reported explicitly; neither is chosen using test labels. Intervals are pointwise paired percentile bootstrap intervals, not multiplicity-adjusted.

| Model | Dataset | Target | Method | N | AUC (95% CI) | PCC continuous | HIDE − baseline AUC (95% CI) |
|---|---|---|---|---:|---|---:|---|
| gemma-2-9b | SQuAD | sentence_similarity | HIDE_score | 2986 | 0.7664 [0.7370, 0.7953] | 0.3873 | 0.0000 [0.0000, 0.0000] |
| gemma-2-9b | SQuAD | sentence_similarity | Omega | 2986 | 0.8468 [0.8249, 0.8677] | 0.4485 | -0.0804 [-0.1048, -0.0559] |
| gemma-2-9b | SQuAD | sentence_similarity | Delta_in | 2986 | 0.5937 [0.5615, 0.6269] | 0.1730 | 0.1727 [0.1249, 0.2174] |
| gemma-2-9b | SQuAD | sentence_similarity | negative_Delta_in | 2986 | 0.4063 [0.3731, 0.4385] | -0.1730 | 0.3601 [0.3164, 0.4045] |
| gemma-2-9b | SQuAD | sentence_similarity | negative_mnll | 2986 | 0.6438 [0.6180, 0.6686] | 0.1009 | 0.1226 [0.0801, 0.1663] |
| gemma-2-9b | SQuAD | sentence_similarity | negative_energy | 2986 | 0.3019 [0.2702, 0.3331] | -0.1290 | 0.4644 [0.4265, 0.5055] |
| gemma-2-9b | SQuAD | sentence_similarity | constant_kernel_control | 2986 | 0.7632 [0.7354, 0.7914] | 0.3868 | 0.0032 [-0.0033, 0.0096] |
| gemma-2-9b | SQuAD | sentence_similarity | negative_output_length | 2986 | 0.8122 [0.7862, 0.8362] | 0.3261 | -0.0458 [-0.0612, -0.0307] |
| gemma-2-9b | SQuAD | rouge_l | HIDE_score | 2986 | 0.7345 [0.7101, 0.7584] | 0.3937 | 0.0000 [0.0000, 0.0000] |
| gemma-2-9b | SQuAD | rouge_l | Omega | 2986 | 0.8310 [0.8143, 0.8478] | 0.5367 | -0.0965 [-0.1160, -0.0781] |
| gemma-2-9b | SQuAD | rouge_l | Delta_in | 2986 | 0.6002 [0.5723, 0.6259] | 0.1469 | 0.1344 [0.0983, 0.1691] |
| gemma-2-9b | SQuAD | rouge_l | negative_Delta_in | 2986 | 0.3998 [0.3741, 0.4277] | -0.1469 | 0.3347 [0.2969, 0.3709] |
| gemma-2-9b | SQuAD | rouge_l | negative_mnll | 2986 | 0.6214 [0.5983, 0.6436] | 0.1469 | 0.1132 [0.0758, 0.1499] |
| gemma-2-9b | SQuAD | rouge_l | negative_energy | 2986 | 0.3517 [0.3250, 0.3776] | -0.1973 | 0.3828 [0.3501, 0.4159] |
| gemma-2-9b | SQuAD | rouge_l | constant_kernel_control | 2986 | 0.7309 [0.7069, 0.7527] | 0.3926 | 0.0036 [-0.0024, 0.0099] |
| gemma-2-9b | SQuAD | rouge_l | negative_output_length | 2986 | 0.7822 [0.7600, 0.8016] | 0.4019 | -0.0477 [-0.0597, -0.0361] |
| gemma-2-9b | SQuAD | exact_match | HIDE_score | 2986 | 0.8108 [0.7769, 0.8431] | 0.2080 | 0.0000 [0.0000, 0.0000] |
| gemma-2-9b | SQuAD | exact_match | Omega | 2986 | 0.9049 [0.8802, 0.9283] | 0.2073 | -0.0941 [-0.1252, -0.0613] |
| gemma-2-9b | SQuAD | exact_match | Delta_in | 2986 | 0.5783 [0.5274, 0.6278] | 0.0323 | 0.2325 [0.1715, 0.2942] |
| gemma-2-9b | SQuAD | exact_match | negative_Delta_in | 2986 | 0.4217 [0.3722, 0.4726] | -0.0323 | 0.3891 [0.3297, 0.4486] |
| gemma-2-9b | SQuAD | exact_match | negative_mnll | 2986 | 0.7007 [0.6691, 0.7323] | 0.1223 | 0.1101 [0.0570, 0.1644] |
| gemma-2-9b | SQuAD | exact_match | negative_energy | 2986 | 0.2511 [0.2048, 0.2952] | -0.1731 | 0.5597 [0.5045, 0.6157] |
| gemma-2-9b | SQuAD | exact_match | constant_kernel_control | 2986 | 0.8032 [0.7694, 0.8367] | 0.2076 | 0.0076 [-0.0006, 0.0161] |
| gemma-2-9b | SQuAD | exact_match | negative_output_length | 2986 | 0.8711 [0.8397, 0.8989] | 0.1066 | -0.0603 [-0.0854, -0.0355] |
| gemma-2-9b | nq_open | sentence_similarity | HIDE_score | 3278 | 0.9054 [0.7379, 0.9881] | 0.1448 | 0.0000 [0.0000, 0.0000] |
| gemma-2-9b | nq_open | sentence_similarity | Omega | 3278 | 0.6563 [0.2249, 0.9853] | 0.0193 | 0.2490 [-0.0687, 0.7586] |
| gemma-2-9b | nq_open | sentence_similarity | Delta_in | 3278 | 0.6785 [0.0110, 0.9356] | 0.0609 | 0.2269 [-0.1704, 0.9008] |
| gemma-2-9b | nq_open | sentence_similarity | negative_Delta_in | 3278 | 0.3215 [0.0644, 0.9890] | -0.0609 | 0.5838 [-0.0720, 0.9212] |
| gemma-2-9b | nq_open | sentence_similarity | negative_mnll | 3278 | 0.5361 [0.1913, 0.9832] | 0.0783 | 0.3692 [-0.0665, 0.6140] |
| gemma-2-9b | nq_open | sentence_similarity | negative_energy | 3278 | 0.5582 [0.2569, 0.8184] | -0.0050 | 0.3472 [0.0203, 0.6577] |
| gemma-2-9b | nq_open | sentence_similarity | constant_kernel_control | 3278 | 0.8857 [0.7044, 0.9863] | 0.1451 | 0.0197 [-0.0018, 0.0432] |
| gemma-2-9b | nq_open | sentence_similarity | negative_output_length | 3278 | 0.9601 [0.8617, 0.9994] | 0.0260 | -0.0548 [-0.1231, -0.0024] |
| gemma-2-9b | nq_open | rouge_l | HIDE_score | 3278 | 0.7228 [0.5125, 0.9185] | 0.0665 | 0.0000 [0.0000, 0.0000] |
| gemma-2-9b | nq_open | rouge_l | Omega | 3278 | 0.4597 [0.3951, 0.5062] | 0.1041 | 0.2631 [0.0141, 0.5213] |
| gemma-2-9b | nq_open | rouge_l | Delta_in | 3278 | 0.6793 [0.5432, 0.9112] | 0.1204 | 0.0436 [-0.1806, 0.3399] |
| gemma-2-9b | nq_open | rouge_l | negative_Delta_in | 3278 | 0.3207 [0.0888, 0.4568] | -0.1204 | 0.4021 [0.0589, 0.6467] |
| gemma-2-9b | nq_open | rouge_l | negative_mnll | 3278 | 0.5133 [0.1822, 0.9677] | 0.1773 | 0.2096 [-0.0528, 0.5538] |
| gemma-2-9b | nq_open | rouge_l | negative_energy | 3278 | 0.4037 [0.1431, 0.7204] | -0.0519 | 0.3192 [0.0095, 0.7743] |
| gemma-2-9b | nq_open | rouge_l | constant_kernel_control | 3278 | 0.7052 [0.4679, 0.9442] | 0.0661 | 0.0177 [-0.0273, 0.0484] |
| gemma-2-9b | nq_open | rouge_l | negative_output_length | 3278 | 0.8159 [0.6554, 0.9297] | 0.0897 | -0.0930 [-0.1478, -0.0096] |
| gemma-2-9b | nq_open | exact_match | HIDE_score | 3278 | undefined | nan | undefined |
| gemma-2-9b | nq_open | exact_match | Omega | 3278 | undefined | nan | undefined |
| gemma-2-9b | nq_open | exact_match | Delta_in | 3278 | undefined | nan | undefined |
| gemma-2-9b | nq_open | exact_match | negative_Delta_in | 3278 | undefined | nan | undefined |
| gemma-2-9b | nq_open | exact_match | negative_mnll | 3278 | undefined | nan | undefined |
| gemma-2-9b | nq_open | exact_match | negative_energy | 3278 | undefined | nan | undefined |
| gemma-2-9b | nq_open | exact_match | constant_kernel_control | 3278 | undefined | nan | undefined |
| gemma-2-9b | nq_open | exact_match | negative_output_length | 3278 | undefined | nan | undefined |
| gemma-2-9b | race | sentence_similarity | HIDE_score | 711 | 0.5852 [0.5417, 0.6272] | 0.1170 | 0.0000 [0.0000, 0.0000] |
| gemma-2-9b | race | sentence_similarity | Omega | 711 | 0.5736 [0.5298, 0.6151] | 0.1560 | 0.0116 [-0.0379, 0.0576] |
| gemma-2-9b | race | sentence_similarity | Delta_in | 711 | 0.5274 [0.4856, 0.5720] | 0.0490 | 0.0579 [-0.0053, 0.1170] |
| gemma-2-9b | race | sentence_similarity | negative_Delta_in | 711 | 0.4726 [0.4280, 0.5144] | -0.0490 | 0.1126 [0.0513, 0.1717] |
| gemma-2-9b | race | sentence_similarity | negative_mnll | 711 | 0.6109 [0.5716, 0.6507] | 0.2008 | -0.0257 [-0.0942, 0.0425] |
| gemma-2-9b | race | sentence_similarity | negative_energy | 711 | 0.5011 [0.4610, 0.5435] | -0.0261 | 0.0841 [0.0190, 0.1443] |
| gemma-2-9b | race | sentence_similarity | constant_kernel_control | 711 | 0.5792 [0.5373, 0.6203] | 0.1170 | 0.0060 [-0.0023, 0.0137] |
| gemma-2-9b | race | sentence_similarity | negative_output_length | 711 | 0.5788 [0.5366, 0.6204] | 0.1255 | 0.0065 [-0.0132, 0.0256] |
| gemma-2-9b | race | rouge_l | HIDE_score | 711 | 0.5758 [0.5297, 0.6170] | 0.1073 | 0.0000 [0.0000, 0.0000] |
| gemma-2-9b | race | rouge_l | Omega | 711 | 0.5670 [0.5228, 0.6088] | 0.1375 | 0.0088 [-0.0403, 0.0560] |
| gemma-2-9b | race | rouge_l | Delta_in | 711 | 0.5287 [0.4855, 0.5731] | 0.0507 | 0.0471 [-0.0148, 0.1062] |
| gemma-2-9b | race | rouge_l | negative_Delta_in | 711 | 0.4713 [0.4269, 0.5145] | -0.0507 | 0.1045 [0.0404, 0.1634] |
| gemma-2-9b | race | rouge_l | negative_mnll | 711 | 0.5998 [0.5592, 0.6411] | 0.2005 | -0.0240 [-0.0937, 0.0443] |
| gemma-2-9b | race | rouge_l | negative_energy | 711 | 0.4957 [0.4550, 0.5391] | -0.0147 | 0.0801 [0.0158, 0.1384] |
| gemma-2-9b | race | rouge_l | constant_kernel_control | 711 | 0.5705 [0.5253, 0.6111] | 0.1076 | 0.0053 [-0.0030, 0.0128] |
| gemma-2-9b | race | rouge_l | negative_output_length | 711 | 0.5641 [0.5211, 0.6054] | 0.1271 | 0.0117 [-0.0077, 0.0313] |
| gemma-2-9b | race | exact_match | HIDE_score | 711 | 0.5986 [0.5559, 0.6394] | 0.1808 | 0.0000 [0.0000, 0.0000] |
| gemma-2-9b | race | exact_match | Omega | 711 | 0.5967 [0.5536, 0.6393] | 0.2263 | 0.0019 [-0.0474, 0.0488] |
| gemma-2-9b | race | exact_match | Delta_in | 711 | 0.5330 [0.4916, 0.5773] | 0.0601 | 0.0656 [0.0026, 0.1230] |
| gemma-2-9b | race | exact_match | negative_Delta_in | 711 | 0.4670 [0.4227, 0.5084] | -0.0601 | 0.1316 [0.0730, 0.1919] |
| gemma-2-9b | race | exact_match | negative_mnll | 711 | 0.6179 [0.5774, 0.6578] | 0.2072 | -0.0193 [-0.0883, 0.0479] |
| gemma-2-9b | race | exact_match | negative_energy | 711 | 0.5264 [0.4846, 0.5687] | 0.0378 | 0.0722 [0.0097, 0.1311] |
| gemma-2-9b | race | exact_match | constant_kernel_control | 711 | 0.5922 [0.5501, 0.6314] | 0.1804 | 0.0064 [-0.0020, 0.0141] |
| gemma-2-9b | race | exact_match | negative_output_length | 711 | 0.5929 [0.5516, 0.6340] | 0.2279 | 0.0057 [-0.0136, 0.0248] |
| gemma-2-9b | triviaqa | sentence_similarity | HIDE_score | 8443 | 0.7686 [0.6338, 0.8980] | 0.3224 | 0.0000 [0.0000, 0.0000] |
| gemma-2-9b | triviaqa | sentence_similarity | Omega | 8443 | 0.6875 [0.3402, 0.9874] | 0.2486 | 0.0811 [-0.2239, 0.3834] |
| gemma-2-9b | triviaqa | sentence_similarity | Delta_in | 8443 | 0.3833 [0.0406, 0.7088] | 0.0815 | 0.3853 [0.1295, 0.6653] |
| gemma-2-9b | triviaqa | sentence_similarity | negative_Delta_in | 8443 | 0.6167 [0.2912, 0.9594] | -0.0815 | 0.1519 [-0.2544, 0.5552] |
| gemma-2-9b | triviaqa | sentence_similarity | negative_mnll | 8443 | 0.7254 [0.3856, 0.9599] | 0.0322 | 0.0432 [-0.3252, 0.4738] |
| gemma-2-9b | triviaqa | sentence_similarity | negative_energy | 8443 | 0.3279 [0.0628, 0.6423] | 0.0059 | 0.4407 [0.1644, 0.6925] |
| gemma-2-9b | triviaqa | sentence_similarity | constant_kernel_control | 8443 | 0.7466 [0.6136, 0.8748] | 0.3242 | 0.0220 [0.0146, 0.0285] |
| gemma-2-9b | triviaqa | sentence_similarity | negative_output_length | 8443 | 0.8655 [0.6821, 0.9935] | 0.1821 | -0.0969 [-0.2084, 0.0194] |
| gemma-2-9b | triviaqa | rouge_l | HIDE_score | 8443 | 0.6028 [0.4714, 0.7272] | 0.1364 | 0.0000 [0.0000, 0.0000] |
| gemma-2-9b | triviaqa | rouge_l | Omega | 8443 | 0.7567 [0.6280, 0.8741] | 0.2192 | -0.1540 [-0.3024, -0.0066] |
| gemma-2-9b | triviaqa | rouge_l | Delta_in | 8443 | 0.3966 [0.2814, 0.5302] | 0.0844 | 0.2061 [0.0492, 0.3660] |
| gemma-2-9b | triviaqa | rouge_l | negative_Delta_in | 8443 | 0.6034 [0.4698, 0.7186] | -0.0844 | -0.0006 [-0.1827, 0.1913] |
| gemma-2-9b | triviaqa | rouge_l | negative_mnll | 8443 | 0.7774 [0.7043, 0.8452] | 0.1708 | -0.1746 [-0.3101, -0.0435] |
| gemma-2-9b | triviaqa | rouge_l | negative_energy | 8443 | 0.2501 [0.1600, 0.3546] | -0.0985 | 0.3526 [0.2301, 0.4634] |
| gemma-2-9b | triviaqa | rouge_l | constant_kernel_control | 8443 | 0.5821 [0.4511, 0.7100] | 0.1344 | 0.0207 [-0.0044, 0.0412] |
| gemma-2-9b | triviaqa | rouge_l | negative_output_length | 8443 | 0.6583 [0.5164, 0.7926] | 0.1437 | -0.0555 [-0.1234, 0.0124] |
| gemma-2-9b | triviaqa | exact_match | HIDE_score | 8443 | undefined | nan | undefined |
| gemma-2-9b | triviaqa | exact_match | Omega | 8443 | undefined | nan | undefined |
| gemma-2-9b | triviaqa | exact_match | Delta_in | 8443 | undefined | nan | undefined |
| gemma-2-9b | triviaqa | exact_match | negative_Delta_in | 8443 | undefined | nan | undefined |
| gemma-2-9b | triviaqa | exact_match | negative_mnll | 8443 | undefined | nan | undefined |
| gemma-2-9b | triviaqa | exact_match | negative_energy | 8443 | undefined | nan | undefined |
| gemma-2-9b | triviaqa | exact_match | constant_kernel_control | 8443 | undefined | nan | undefined |
| gemma-2-9b | triviaqa | exact_match | negative_output_length | 8443 | undefined | nan | undefined |
| llama3-8b | SQuAD | sentence_similarity | HIDE_score | 2629 | 0.6852 [0.6385, 0.7297] | 0.3085 | 0.0000 [0.0000, 0.0000] |
| llama3-8b | SQuAD | sentence_similarity | Omega | 2629 | 0.7770 [0.7461, 0.8079] | 0.3319 | -0.0918 [-0.1317, -0.0523] |
| llama3-8b | SQuAD | sentence_similarity | Delta_in | 2629 | 0.4780 [0.4397, 0.5213] | -0.0015 | 0.2072 [0.1483, 0.2585] |
| llama3-8b | SQuAD | sentence_similarity | negative_Delta_in | 2629 | 0.5220 [0.4787, 0.5603] | 0.0015 | 0.1631 [0.0994, 0.2280] |
| llama3-8b | SQuAD | sentence_similarity | negative_mnll | 2629 | 0.6649 [0.6336, 0.6956] | 0.1174 | 0.0203 [-0.0399, 0.0826] |
| llama3-8b | SQuAD | sentence_similarity | negative_energy | 2629 | 0.5854 [0.5489, 0.6237] | 0.0065 | 0.0998 [0.0377, 0.1651] |
| llama3-8b | SQuAD | sentence_similarity | constant_kernel_control | 2629 | 0.6873 [0.6431, 0.7308] | 0.3085 | -0.0021 [-0.0119, 0.0075] |
| llama3-8b | SQuAD | sentence_similarity | negative_output_length | 2629 | 0.6901 [0.6518, 0.7273] | 0.2440 | -0.0049 [-0.0231, 0.0134] |
| llama3-8b | SQuAD | rouge_l | HIDE_score | 2629 | 0.6274 [0.5934, 0.6584] | 0.2582 | 0.0000 [0.0000, 0.0000] |
| llama3-8b | SQuAD | rouge_l | Omega | 2629 | 0.7507 [0.7265, 0.7739] | 0.4216 | -0.1233 [-0.1540, -0.0952] |
| llama3-8b | SQuAD | rouge_l | Delta_in | 2629 | 0.4485 [0.4176, 0.4784] | -0.0690 | 0.1789 [0.1351, 0.2194] |
| llama3-8b | SQuAD | rouge_l | negative_Delta_in | 2629 | 0.5515 [0.5216, 0.5824] | 0.0690 | 0.0759 [0.0295, 0.1203] |
| llama3-8b | SQuAD | rouge_l | negative_mnll | 2629 | 0.6426 [0.6170, 0.6683] | 0.2104 | -0.0152 [-0.0636, 0.0313] |
| llama3-8b | SQuAD | rouge_l | negative_energy | 2629 | 0.5473 [0.5203, 0.5760] | 0.0294 | 0.0801 [0.0324, 0.1277] |
| llama3-8b | SQuAD | rouge_l | constant_kernel_control | 2629 | 0.6345 [0.6027, 0.6640] | 0.2582 | -0.0071 [-0.0151, 0.0011] |
| llama3-8b | SQuAD | rouge_l | negative_output_length | 2629 | 0.6590 [0.6311, 0.6856] | 0.3186 | -0.0316 [-0.0456, -0.0182] |
| llama3-8b | SQuAD | exact_match | HIDE_score | 2629 | 0.8188 [0.7688, 0.8625] | 0.1874 | 0.0000 [0.0000, 0.0000] |
| llama3-8b | SQuAD | exact_match | Omega | 2629 | 0.8637 [0.8256, 0.8981] | 0.1298 | -0.0449 [-0.0938, 0.0038] |
| llama3-8b | SQuAD | exact_match | Delta_in | 2629 | 0.5468 [0.4874, 0.6022] | 0.0030 | 0.2720 [0.1950, 0.3469] |
| llama3-8b | SQuAD | exact_match | negative_Delta_in | 2629 | 0.4532 [0.3978, 0.5126] | -0.0030 | 0.3656 [0.2887, 0.4406] |
| llama3-8b | SQuAD | exact_match | negative_mnll | 2629 | 0.7214 [0.6866, 0.7549] | 0.1100 | 0.0974 [0.0285, 0.1619] |
| llama3-8b | SQuAD | exact_match | negative_energy | 2629 | 0.6509 [0.5894, 0.7108] | 0.0709 | 0.1679 [0.0865, 0.2461] |
| llama3-8b | SQuAD | exact_match | constant_kernel_control | 2629 | 0.8140 [0.7640, 0.8592] | 0.1874 | 0.0048 [-0.0013, 0.0117] |
| llama3-8b | SQuAD | exact_match | negative_output_length | 2629 | 0.8114 [0.7692, 0.8524] | 0.0715 | 0.0073 [-0.0200, 0.0347] |
| llama3-8b | nq_open | sentence_similarity | HIDE_score | 1912 | 0.8486 [0.6466, 0.9558] | 0.1124 | 0.0000 [0.0000, 0.0000] |
| llama3-8b | nq_open | sentence_similarity | Omega | 1912 | 0.8991 [0.7791, 0.9884] | 0.0677 | -0.0505 [-0.1827, 0.0647] |
| llama3-8b | nq_open | sentence_similarity | Delta_in | 1912 | 0.8167 [0.7010, 0.9379] | 0.0333 | 0.0319 [-0.2814, 0.2272] |
| llama3-8b | nq_open | sentence_similarity | negative_Delta_in | 1912 | 0.1833 [0.0621, 0.2990] | -0.0333 | 0.6653 [0.5409, 0.7937] |
| llama3-8b | nq_open | sentence_similarity | negative_mnll | 1912 | 0.6758 [0.5245, 0.8837] | 0.0990 | 0.1728 [-0.0045, 0.3805] |
| llama3-8b | nq_open | sentence_similarity | negative_energy | 1912 | 0.7772 [0.5271, 0.9349] | 0.1136 | 0.0714 [-0.0001, 0.1566] |
| llama3-8b | nq_open | sentence_similarity | constant_kernel_control | 1912 | 0.8372 [0.6402, 0.9461] | 0.1124 | 0.0114 [0.0001, 0.0236] |
| llama3-8b | nq_open | sentence_similarity | negative_output_length | 1912 | 0.8411 [0.6097, 0.9992] | 0.0262 | 0.0076 [-0.0690, 0.0847] |
| llama3-8b | nq_open | rouge_l | HIDE_score | 1912 | 0.7977 [0.5128, 0.9355] | 0.0801 | 0.0000 [0.0000, 0.0000] |
| llama3-8b | nq_open | rouge_l | Omega | 1912 | 0.7691 [0.5481, 0.9518] | 0.0916 | 0.0286 [-0.3562, 0.3517] |
| llama3-8b | nq_open | rouge_l | Delta_in | 1912 | 0.6487 [0.5146, 0.7690] | 0.0467 | 0.1489 [-0.0666, 0.3333] |
| llama3-8b | nq_open | rouge_l | negative_Delta_in | 1912 | 0.3513 [0.2310, 0.4854] | -0.0467 | 0.4464 [0.0649, 0.6735] |
| llama3-8b | nq_open | rouge_l | negative_mnll | 1912 | 0.6973 [0.5271, 0.8760] | 0.0883 | 0.1003 [-0.2986, 0.4013] |
| llama3-8b | nq_open | rouge_l | negative_energy | 1912 | 0.7339 [0.5283, 0.9212] | 0.0871 | 0.0638 [-0.3078, 0.3835] |
| llama3-8b | nq_open | rouge_l | constant_kernel_control | 1912 | 0.7903 [0.5102, 0.9296] | 0.0801 | 0.0074 [-0.0075, 0.0213] |
| llama3-8b | nq_open | rouge_l | negative_output_length | 1912 | 0.8756 [0.7497, 0.9742] | 0.1197 | -0.0780 [-0.2675, 0.0541] |
| llama3-8b | nq_open | exact_match | HIDE_score | 1912 | undefined | nan | undefined |
| llama3-8b | nq_open | exact_match | Omega | 1912 | undefined | nan | undefined |
| llama3-8b | nq_open | exact_match | Delta_in | 1912 | undefined | nan | undefined |
| llama3-8b | nq_open | exact_match | negative_Delta_in | 1912 | undefined | nan | undefined |
| llama3-8b | nq_open | exact_match | negative_mnll | 1912 | undefined | nan | undefined |
| llama3-8b | nq_open | exact_match | negative_energy | 1912 | undefined | nan | undefined |
| llama3-8b | nq_open | exact_match | constant_kernel_control | 1912 | undefined | nan | undefined |
| llama3-8b | nq_open | exact_match | negative_output_length | 1912 | undefined | nan | undefined |
| llama3-8b | race | sentence_similarity | HIDE_score | 1553 | 0.7648 [0.7371, 0.7928] | 0.2283 | 0.0000 [0.0000, 0.0000] |
| llama3-8b | race | sentence_similarity | Omega | 1553 | 0.7983 [0.7725, 0.8229] | 0.2452 | -0.0335 [-0.0550, -0.0127] |
| llama3-8b | race | sentence_similarity | Delta_in | 1553 | 0.6287 [0.5938, 0.6623] | 0.0919 | 0.1361 [0.0943, 0.1759] |
| llama3-8b | race | sentence_similarity | negative_Delta_in | 1553 | 0.3713 [0.3377, 0.4062] | -0.0919 | 0.3935 [0.3451, 0.4408] |
| llama3-8b | race | sentence_similarity | negative_mnll | 1553 | 0.5742 [0.5394, 0.6084] | 0.1153 | 0.1906 [0.1465, 0.2360] |
| llama3-8b | race | sentence_similarity | negative_energy | 1553 | 0.5854 [0.5501, 0.6195] | 0.1812 | 0.1794 [0.1363, 0.2236] |
| llama3-8b | race | sentence_similarity | constant_kernel_control | 1553 | 0.7649 [0.7383, 0.7929] | 0.2283 | -0.0001 [-0.0050, 0.0052] |
| llama3-8b | race | sentence_similarity | negative_output_length | 1553 | 0.7871 [0.7622, 0.8120] | 0.2043 | -0.0224 [-0.0340, -0.0116] |
| llama3-8b | race | rouge_l | HIDE_score | 1553 | 0.7423 [0.7160, 0.7692] | 0.4093 | 0.0000 [0.0000, 0.0000] |
| llama3-8b | race | rouge_l | Omega | 1553 | 0.7719 [0.7488, 0.7945] | 0.5098 | -0.0296 [-0.0489, -0.0104] |
| llama3-8b | race | rouge_l | Delta_in | 1553 | 0.5875 [0.5547, 0.6180] | 0.2015 | 0.1548 [0.1180, 0.1939] |
| llama3-8b | race | rouge_l | negative_Delta_in | 1553 | 0.4125 [0.3820, 0.4453] | -0.2015 | 0.3299 [0.2842, 0.3724] |
| llama3-8b | race | rouge_l | negative_mnll | 1553 | 0.5469 [0.5154, 0.5784] | 0.0657 | 0.1955 [0.1527, 0.2371] |
| llama3-8b | race | rouge_l | negative_energy | 1553 | 0.5564 [0.5253, 0.5879] | 0.1092 | 0.1860 [0.1460, 0.2277] |
| llama3-8b | race | rouge_l | constant_kernel_control | 1553 | 0.7411 [0.7150, 0.7672] | 0.4093 | 0.0013 [-0.0041, 0.0067] |
| llama3-8b | race | rouge_l | negative_output_length | 1553 | 0.7736 [0.7507, 0.7969] | 0.4650 | -0.0313 [-0.0422, -0.0210] |
| llama3-8b | race | exact_match | HIDE_score | 1553 | 0.8238 [0.7979, 0.8488] | 0.3757 | 0.0000 [0.0000, 0.0000] |
| llama3-8b | race | exact_match | Omega | 1553 | 0.8717 [0.8517, 0.8900] | 0.2878 | -0.0478 [-0.0713, -0.0241] |
| llama3-8b | race | exact_match | Delta_in | 1553 | 0.6851 [0.6446, 0.7237] | 0.1953 | 0.1387 [0.0900, 0.1868] |
| llama3-8b | race | exact_match | negative_Delta_in | 1553 | 0.3149 [0.2763, 0.3554] | -0.1953 | 0.5089 [0.4636, 0.5528] |
| llama3-8b | race | exact_match | negative_mnll | 1553 | 0.7080 [0.6784, 0.7354] | 0.2059 | 0.1158 [0.0707, 0.1601] |
| llama3-8b | race | exact_match | negative_energy | 1553 | 0.6719 [0.6338, 0.7113] | 0.1811 | 0.1519 [0.1023, 0.1966] |
| llama3-8b | race | exact_match | constant_kernel_control | 1553 | 0.8190 [0.7917, 0.8450] | 0.3757 | 0.0048 [0.0004, 0.0096] |
| llama3-8b | race | exact_match | negative_output_length | 1553 | 0.8455 [0.8228, 0.8673] | 0.2155 | -0.0216 [-0.0309, -0.0129] |
| llama3-8b | triviaqa | sentence_similarity | HIDE_score | 2755 | 0.9234 [0.8295, 0.9880] | 0.2559 | 0.0000 [0.0000, 0.0000] |
| llama3-8b | triviaqa | sentence_similarity | Omega | 2755 | 0.9731 [0.9531, 0.9881] | 0.0469 | -0.0497 [-0.1269, 0.0065] |
| llama3-8b | triviaqa | sentence_similarity | Delta_in | 2755 | 0.9845 [0.9699, 0.9961] | 0.0656 | -0.0611 [-0.1557, -0.0007] |
| llama3-8b | triviaqa | sentence_similarity | negative_Delta_in | 2755 | 0.0155 [0.0039, 0.0301] | -0.0656 | 0.9080 [0.8149, 0.9760] |
| llama3-8b | triviaqa | sentence_similarity | negative_mnll | 2755 | 0.2579 [0.1367, 0.4104] | 0.0365 | 0.6656 [0.4692, 0.8319] |
| llama3-8b | triviaqa | sentence_similarity | negative_energy | 2755 | 0.2784 [0.1817, 0.3966] | -0.0694 | 0.6450 [0.5030, 0.7661] |
| llama3-8b | triviaqa | sentence_similarity | constant_kernel_control | 2755 | 0.9266 [0.8406, 0.9862] | 0.2559 | -0.0032 [-0.0129, 0.0050] |
| llama3-8b | triviaqa | sentence_similarity | negative_output_length | 2755 | 0.9742 [0.9576, 0.9877] | 0.0649 | -0.0508 [-0.1311, 0.0044] |
| llama3-8b | triviaqa | rouge_l | HIDE_score | 2755 | 0.9031 [0.8158, 0.9715] | 0.2651 | 0.0000 [0.0000, 0.0000] |
| llama3-8b | triviaqa | rouge_l | Omega | 2755 | 0.9729 [0.9570, 0.9862] | 0.5212 | -0.0699 [-0.1459, -0.0096] |
| llama3-8b | triviaqa | rouge_l | Delta_in | 2755 | 0.9855 [0.9759, 0.9940] | 0.5397 | -0.0825 [-0.1729, -0.0117] |
| llama3-8b | triviaqa | rouge_l | negative_Delta_in | 2755 | 0.0145 [0.0060, 0.0241] | -0.5397 | 0.8886 [0.8027, 0.9576] |
| llama3-8b | triviaqa | rouge_l | negative_mnll | 2755 | 0.2569 [0.1760, 0.3599] | -0.2803 | 0.6462 [0.4881, 0.7761] |
| llama3-8b | triviaqa | rouge_l | negative_energy | 2755 | 0.3755 [0.2714, 0.4856] | -0.2213 | 0.5276 [0.3730, 0.6681] |
| llama3-8b | triviaqa | rouge_l | constant_kernel_control | 2755 | 0.9049 [0.8204, 0.9703] | 0.2651 | -0.0018 [-0.0084, 0.0036] |
| llama3-8b | triviaqa | rouge_l | negative_output_length | 2755 | 0.9742 [0.9592, 0.9862] | 0.4995 | -0.0711 [-0.1463, -0.0114] |
| llama3-8b | triviaqa | exact_match | HIDE_score | 2755 | 0.9459 [0.8486, 0.9946] | 0.0819 | 0.0000 [0.0000, 0.0000] |
| llama3-8b | triviaqa | exact_match | Omega | 2755 | 0.9757 [0.9644, 0.9873] | 0.0903 | -0.0298 [-0.1369, 0.0282] |
| llama3-8b | triviaqa | exact_match | Delta_in | 2755 | 0.9898 [0.9790, 0.9967] | 0.1159 | -0.0440 [-0.1450, 0.0138] |
| llama3-8b | triviaqa | exact_match | negative_Delta_in | 2755 | 0.0102 [0.0033, 0.0210] | -0.1159 | 0.9357 [0.8417, 0.9891] |
| llama3-8b | triviaqa | exact_match | negative_mnll | 2755 | 0.4503 [0.1845, 0.7202] | 0.0059 | 0.4955 [0.1305, 0.8079] |
| llama3-8b | triviaqa | exact_match | negative_energy | 2755 | 0.3651 [0.1498, 0.5707] | -0.0091 | 0.5808 [0.2814, 0.8426] |
| llama3-8b | triviaqa | exact_match | constant_kernel_control | 2755 | 0.9529 [0.8734, 0.9927] | 0.0819 | -0.0071 [-0.0256, 0.0031] |
| llama3-8b | triviaqa | exact_match | negative_output_length | 2755 | 0.9839 [0.9657, 0.9964] | 0.0799 | -0.0380 [-0.1189, 0.0056] |
