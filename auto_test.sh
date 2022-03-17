# set -x
# set -e
# executor always
# source ~/cbc3env/bin/activate

# build_names
# "./alternate.x86_64"
# "./city.x86_64"
# "./office.x86_64"
# "staticasl"
# "cathedral"
# "gallery"
# "kozehd"

# difficulties
# "easiest"
# "hardest"
# "progressive"
# ------------------------------------------------------------------------

# N3D
# 1) does train_time perf translate to test_time?
# 2) which models xtest best on gallery, cathedral, kozehdrs?
# 3) does a R model trained on SC do well in R?
# 4) does scope analysis also hold up at test_time?
# 5) dreamer vs n3d tests in S, R, and K
# 6) best in each env

# 1)
# S_E2E S
# S_SCR S
# 3)
# SC_R R

# 4)
# S_E2E SCR
# SC_E2E SCR
# SCR_E2E SCR
# R_E2E SCR
# S_SCR SCR
# SC_SCR SCR
# SCR_SCR SCR
# R_SCR SCR

# 
    # e2e
#   "~/navdreams_data/results/models/gym/navrep3daltenv_2021_11_01__08_52_03_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" \
    # n3d
#   "~/navdreams_data/results/models/gym/navrep3daslencodedenv_2021_12_11__00_23_55_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
#   "~/navdreams_data/results/models/gym/navrep3daslencodedenv_2021_12_08__10_18_09_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
#   "~/navdreams_data/results/models/gym/navrep3dSCRencodedenv_2021_12_12__16_46_51_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
#   "~/navdreams_data/results/models/gym/navrep3dkozehdencodedenv_2022_01_17__12_55_53_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
#   "~/navdreams_data/results/models/gym/navrep3dkozehdencodedenv_2022_01_17__12_55_53_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
#   "~/navdreams_data/results/models/gym/navrep3dkozehdoldencodedenv_2022_01_13__16_13_45_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
#   "~/navdreams_data/results/models/gym/navrep3dkozehdoldencodedenv_2022_01_13__16_13_45_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
#   "~/navdreams_data/results/models/gym/navrep3dkozehdremptyencodedenv_2022_01_30__22_12_30_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCRK_bestckpt.zip" \
#   "~/navdreams_data/results/models/gym/navrep3dkozehdremptyencodedenv_2022_01_30__22_12_30_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCRK_ckpt.zip" \
#   "~/navdreams_data/results/models/gym/navrep3daltencodedenv_2021_12_15__08_43_12_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
#   "~/navdreams_data/results/models/gym/navrep3daslfixedencodedenv_2021_12_29__17_17_16_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
    # untested
#   "~/navdreams_data/results/models/gym/navrep3dkozehdrencodedenv_2022_01_28__06_44_50_DISCRETE_PPO_GPT_V_ONLY_V64M64_K2_ckpt.zip" \
#   "~/navdreams_data/results/models/gym/navrep3dkozehdrencodedenv_2022_01_25__21_34_44_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCRK_ckpt.zip" \
#   "~/navdreams_data/results/models/gym/navrep3dkozehdrencodedenv_2022_01_26__08_56_54_DISCRETE_PPO_GPT_V_ONLY_V64M64_K_ckpt.zip" \
#   "~/navdreams_data/results/models/gym/navrep3dkozehdrsencodedenv_2022_02_02__17_18_59_DISCRETE_PPO_GPT_V_ONLY_V64M64_K2_bestckpt.zip" \
# ------------------------------------------------------------------------

# 5)
###  cd ~/Code/cbsim/navrep3d
###  python test_any.py --model-path \
###    "~/navdreams_data/results/models/gym/navrep3daltencodedenv_2021_12_15__08_43_12_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
###    --n-episodes $1 --build-name "./alternate.x86_64" --difficulty-mode "hardest" --render False
###  python test_any.py --model-path \
###    "~/navdreams_data/results/models/gym/navrep3daslfixedencodedenv_2021_12_29__17_17_16_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
###    --n-episodes $1 --build-name "staticasl" --difficulty-mode "medium" --render False
###  python test_any.py --model-path \
###    "~/navdreams_data/results/models/gym/navrep3dkozehdrsencodedenv_2022_02_02__17_18_59_DISCRETE_PPO_GPT_V_ONLY_V64M64_K2_bestckpt.zip" \
###    --n-episodes $1 --build-name "kozehd" --difficulty-mode "easiest" --render False
###  python test_any.py --model-path \
###    "~/navdreams_data/results/models/gym/navrep3dkozehdrsencodedenv_2022_02_02__17_18_59_DISCRETE_PPO_GPT_V_ONLY_V64M64_K2_bestckpt.zip" \
###    --n-episodes $1 --build-name "kozehd" --difficulty-mode "easy" --render False
###  cd ~/Code/pydreamer
###  python test.py --run-id "f3f47a18b9334a4baa97c728143a00c6" --n-episodes $1 --build-name "./alternate.x86_64" --difficulty-mode "hardest" --render False
###  python test.py --run-id "a1ec5269279f46f79af2884526590592" --n-episodes $1 --build-name "staticasl" --difficulty-mode "medium" --render False
###  python test.py --run-id "3aaa8d09bce64dd888240a04b714aec7" --n-episodes $1 --build-name "kozehd" --difficulty-mode "easiest" --render False
###  python test.py --run-id "3aaa8d09bce64dd888240a04b714aec7" --n-episodes $1 --build-name "kozehd" --difficulty-mode "easy" --render False
###  cd ~/Code/cbsim/navrep3d
###  python test_any.py --model-path \
###    "~/navdreams_data/results/models/gym/navrep3daltenv_2021_11_15__16_16_40_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" \
###    --n-episodes $1 --build-name "./alternate.x86_64" --difficulty-mode "hardest" --render False
###  python test_any.py --model-path \
###    "~/navdreams_data/results/models/gym/navrep3daslfixedencodedenv_2022_01_01__13_09_23_DISCRETE_PPO_E2E_VCARCH_C64_bestckpt.zip" \
###    --n-episodes $1 --build-name "staticasl" --difficulty-mode "medium" --render False
###  python test_any.py --model-path \
###    "~/navdreams_data/results/models/gym/navrep3dkozehdrsencodedenv_2022_02_06__22_58_00_DISCRETE_PPO_E2E_VCARCH_C64_bestckpt.zip" \
###    --n-episodes $1 --build-name "kozehd" --difficulty-mode "easiest" --render False
###  python test_any.py --model-path \
###    "~/navdreams_data/results/models/gym/navrep3dkozehdrsencodedenv_2022_02_06__22_58_00_DISCRETE_PPO_E2E_VCARCH_C64_bestckpt.zip" \
###    --n-episodes $1 --build-name "kozehd" --difficulty-mode "easy" --render False
###  # 2)
###  python test_any.py --model-path \
###    "~/navdreams_data/results/models/gym/navrep3daltencodedenv_2021_12_15__08_43_12_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###    --n-episodes $1 --build-name "gallery" --difficulty-mode "easy" --render False
###  python test_any.py --model-path \
###    "~/navdreams_data/results/models/gym/navrep3dSCencodedenv_2021_12_10__19_33_51_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###    --n-episodes $1 --build-name "gallery" --difficulty-mode "easy" --render False
###  python test_any.py --model-path \
###    "~/navdreams_data/results/models/gym/navrep3dSCRencodedenv_2021_12_12__16_46_51_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###    --n-episodes $1 --build-name "gallery" --difficulty-mode "easy" --render False
###  python test_any.py --model-path \
###    "~/navdreams_data/results/models/gym/navrep3daltenv_2021_11_15__16_16_40_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" \
###    --n-episodes $1 --build-name "gallery" --difficulty-mode "easy" --render False
###  python test_any.py --model-path \
###    "~/navdreams_data/results/models/gym/navrep3dSCenv_2021_12_16__01_55_07_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" \
###    --n-episodes $1 --build-name "gallery" --difficulty-mode "easy" --render False
###  python test_any.py --model-path \
###    "~/navdreams_data/results/models/gym/navrep3dSCRencodedenv_2021_12_05__13_26_46_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" \
###    --n-episodes $1 --build-name "gallery" --difficulty-mode "easy" --render False
###  # cathedral
###  python test_any.py --model-path \
###    "~/navdreams_data/results/models/gym/navrep3daltencodedenv_2021_12_15__08_43_12_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###    --n-episodes $1 --build-name "cathedral" --difficulty-mode "easy" --render False
###  python test_any.py --model-path \
###    "~/navdreams_data/results/models/gym/navrep3dSCencodedenv_2021_12_10__19_33_51_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###    --n-episodes $1 --build-name "cathedral" --difficulty-mode "easy" --render False
###  python test_any.py --model-path \
###    "~/navdreams_data/results/models/gym/navrep3dSCRencodedenv_2021_12_12__16_46_51_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###    --n-episodes $1 --build-name "cathedral" --difficulty-mode "easy" --render False
###  python test_any.py --model-path \
###    "~/navdreams_data/results/models/gym/navrep3daltenv_2021_11_15__16_16_40_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" \
###    --n-episodes $1 --build-name "cathedral" --difficulty-mode "easy" --render False
###  python test_any.py --model-path \
###    "~/navdreams_data/results/models/gym/navrep3dSCenv_2021_12_16__01_55_07_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" \
###    --n-episodes $1 --build-name "cathedral" --difficulty-mode "easy" --render False
###  python test_any.py --model-path \
###    "~/navdreams_data/results/models/gym/navrep3dSCRencodedenv_2021_12_05__13_26_46_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" \
###    --n-episodes $1 --build-name "cathedral" --difficulty-mode "easy" --render False
# special test - figure out where this goes
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dcathedralencodedenv_2022_02_14__10_22_45_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
###   --n-episodes $1 --build-name "cathedral" --difficulty-mode "easy" --render False
#
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3daltencodedenv_2021_12_15__08_43_12_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###   --n-episodes $1 --build-name "cathedral" --difficulty-mode "medium" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dSCencodedenv_2021_12_10__19_33_51_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###   --n-episodes $1 --build-name "cathedral" --difficulty-mode "medium" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dSCRencodedenv_2021_12_12__16_46_51_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###   --n-episodes $1 --build-name "cathedral" --difficulty-mode "medium" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3daltenv_2021_11_15__16_16_40_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" \
###   --n-episodes $1 --build-name "cathedral" --difficulty-mode "medium" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dSCenv_2021_12_16__01_55_07_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" \
###   --n-episodes $1 --build-name "cathedral" --difficulty-mode "medium" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dSCRencodedenv_2021_12_05__13_26_46_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" \
###   --n-episodes $1 --build-name "cathedral" --difficulty-mode "medium" --render False
# kozehd
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3daltencodedenv_2021_12_15__08_43_12_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###   --n-episodes $1 --build-name "kozehd" --difficulty-mode "easiest" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dSCencodedenv_2021_12_10__19_33_51_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###   --n-episodes $1 --build-name "kozehd" --difficulty-mode "easiest" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dSCRencodedenv_2021_12_12__16_46_51_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###   --n-episodes $1 --build-name "kozehd" --difficulty-mode "easiest" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3daltenv_2021_11_15__16_16_40_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" \
###   --n-episodes $1 --build-name "kozehd" --difficulty-mode "easiest" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dSCenv_2021_12_16__01_55_07_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" \
###   --n-episodes $1 --build-name "kozehd" --difficulty-mode "easiest" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dSCRencodedenv_2021_12_05__13_26_46_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" \
###   --n-episodes $1 --build-name "kozehd" --difficulty-mode "easiest" --render False
### # 1)
### cd ~/Code/cbsim/navrep3d
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3daltencodedenv_2021_12_15__08_43_12_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###   --n-episodes $1 --build-name "./alternate.x86_64" --difficulty-mode "hardest" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3daltenv_2021_11_01__08_52_03_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" \
###   --n-episodes $1 --build-name "./alternate.x86_64" --difficulty-mode "hardest" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3daslfixedencodedenv_2021_12_29__17_17_16_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
###   --n-episodes $1 --build-name "staticasl" --difficulty-mode "medium" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3daslfixedencodedenv_2022_01_01__13_09_23_DISCRETE_PPO_E2E_VCARCH_C64_bestckpt.zip" \
###   --n-episodes $1 --build-name "staticasl" --difficulty-mode "medium" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dcathedralencodedenv_2022_02_14__10_22_45_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
###   --n-episodes $1 --build-name "cathedral" --difficulty-mode "medium" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dcathedralenv_2022_02_11__18_09_16_DISCRETE_PPO_E2E_VCARCH_C64_bestckpt.zip" \
###   --n-episodes $1 --build-name "cathedral" --difficulty-mode "medium" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dgalleryencodedenv_2022_02_11__21_52_34_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
###   --n-episodes $1 --build-name "gallery" --difficulty-mode "easy" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dgalleryenv_2022_02_16__15_08_38_DISCRETE_PPO_E2E_VCARCH_C64_bestckpt.zip" \
###   --n-episodes $1 --build-name "gallery" --difficulty-mode "easy" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dkozehdrsencodedenv_2022_02_02__17_18_59_DISCRETE_PPO_GPT_V_ONLY_V64M64_K2_bestckpt.zip" \
###   --n-episodes $1 --build-name "kozehd" --difficulty-mode "easy" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dkozehdrsencodedenv_2022_02_06__22_46_29_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCRK_bestckpt.zip" \
###   --n-episodes $1 --build-name "kozehd" --difficulty-mode "easy" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dkozehdrsencodedenv_2022_02_06__22_58_00_DISCRETE_PPO_E2E_VCARCH_C64_bestckpt.zip" \
###   --n-episodes $1 --build-name "kozehd" --difficulty-mode "easy" --render False
# 6)
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3daltencodedenv_2021_12_06__21_45_47_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
###   --n-episodes $1 --build-name "./alternate.x86_64" --difficulty-mode "hardest" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3daltencodedenv_2021_12_13__09_05_44_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
###   --n-episodes $1 --build-name "./alternate.x86_64" --difficulty-mode "hardest" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3daltencodedenv_2021_12_15__08_43_12_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
###   --n-episodes $1 --build-name "./alternate.x86_64" --difficulty-mode "hardest" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3daltenv_2021_11_01__08_52_03_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" \
###   --n-episodes $1 --build-name "./alternate.x86_64" --difficulty-mode "hardest" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3daltenv_2021_11_10__12_38_52_DISCRETE_PPO_E2E_VCARCH_C64_bestckpt.zip" \
###   --n-episodes $1 --build-name "./alternate.x86_64" --difficulty-mode "hardest" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3daltenv_2021_11_15__16_16_40_DISCRETE_PPO_E2E_VCARCH_C64_bestckpt.zip" \
###   --n-episodes $1 --build-name "./alternate.x86_64" --difficulty-mode "hardest" --render False
### 
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dcityencodedenv_2022_02_18__18_26_31_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
###   --n-episodes $1 --build-name "./city.x86_64" --difficulty-mode "hardest" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dcityencodedenv_2022_02_21__07_32_02_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
###   --n-episodes $1 --build-name "./city.x86_64" --difficulty-mode "hardest" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dcityencodedenv_2022_02_21__15_00_05_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
###   --n-episodes $1 --build-name "./city.x86_64" --difficulty-mode "hardest" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dcityenv_2022_02_17__21_27_56_DISCRETE_PPO_E2E_VCARCH_C64_bestckpt.zip" \
###   --n-episodes $1 --build-name "./city.x86_64" --difficulty-mode "hardest" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dcityenv_2022_02_19__16_34_05_DISCRETE_PPO_E2E_VCARCH_C64_bestckpt.zip" \
###   --n-episodes $1 --build-name "./city.x86_64" --difficulty-mode "hardest" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dcityenv_2022_02_22__10_41_26_DISCRETE_PPO_E2E_VCARCH_C64_bestckpt.zip" \
###   --n-episodes $1 --build-name "./city.x86_64" --difficulty-mode "hardest" --render False
### 
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dofficeencodedenv_2022_02_17__21_28_36_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
###   --n-episodes $1 --build-name "./office.x86_64" --difficulty-mode "random" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dofficeencodedenv_2022_02_18__18_26_17_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
###   --n-episodes $1 --build-name "./office.x86_64" --difficulty-mode "random" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dofficeencodedenv_2022_02_19__16_33_28_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
###   --n-episodes $1 --build-name "./office.x86_64" --difficulty-mode "random" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dofficeenv_2022_02_17__21_27_47_DISCRETE_PPO_E2E_VCARCH_C64_bestckpt.zip" \
###   --n-episodes $1 --build-name "./office.x86_64" --difficulty-mode "random" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dofficeenv_2022_02_19__14_29_25_DISCRETE_PPO_E2E_VCARCH_C64_bestckpt.zip" \
###   --n-episodes $1 --build-name "./office.x86_64" --difficulty-mode "random" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dofficeenv_2022_02_21__07_31_03_DISCRETE_PPO_E2E_VCARCH_C64_bestckpt.zip" \
###   --n-episodes $1 --build-name "./office.x86_64" --difficulty-mode "random" --render False
### 
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3daslfixedencodedenv_2021_12_29__17_17_16_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
###   --n-episodes $1 --build-name "./staticasl.x86_64" --difficulty-mode "medium" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3daslfixedencodedenv_2022_01_01__13_09_23_DISCRETE_PPO_E2E_VCARCH_C64_bestckpt.zip" \
###   --n-episodes $1 --build-name "./staticasl.x86_64" --difficulty-mode "medium" --render False

# Generalists
# U3-trained C in cathedral (?), gallery easy, and kozehd (easiest?)
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dunity3encodedenv_2022_01_06__22_23_13_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
###   --n-episodes $1 --build-name "cathedral" --difficulty-mode "easy" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dunity3encodedenv_2022_01_06__22_23_13_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
###   --n-episodes $1 --build-name "gallery" --difficulty-mode "easy" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dunity3encodedenv_2022_01_06__22_23_13_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
###   --n-episodes $1 --build-name "kozehd" --difficulty-mode "easiest" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dunity3encodedenv_2022_01_06__22_23_13_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
###   --n-episodes $1 --build-name "./alternate.x86_64" --difficulty-mode "hardest" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dunity3encodedenv_2022_01_06__22_23_13_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
###   --n-episodes $1 --build-name "./city.x86_64" --difficulty-mode "hardest" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dunity3encodedenv_2022_01_06__22_23_13_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
###   --n-episodes $1 --build-name "./office.x86_64" --difficulty-mode "random" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dunity3encodedenv_2022_01_06__22_23_13_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
###   --n-episodes $1 --build-name "staticasl" --difficulty-mode "medium" --render False

# Generalists Win
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dSCRencodedenv_2021_12_06__21_58_01_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###   --n-episodes $1 --build-name "cathedral" --difficulty-mode "medium" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dSCRencodedenv_2021_12_06__21_58_01_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###   --n-episodes $1 --build-name "gallery" --difficulty-mode "easy" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dSCRencodedenv_2021_12_06__21_58_01_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###   --n-episodes $1 --build-name "kozehd" --difficulty-mode "easier" --render False

# Generalists Cost
# SCR-trained C in simple, city, office, and staticasl
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dSCRencodedenv_2021_12_12__16_46_51_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###   --n-episodes $1 --build-name "./alternate.x86_64" --difficulty-mode "hardest" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dSCRencodedenv_2021_12_12__16_46_51_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###   --n-episodes $1 --build-name "./city.x86_64" --difficulty-mode "hardest" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dSCRencodedenv_2021_12_12__16_46_51_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###   --n-episodes $1 --build-name "./office.x86_64" --difficulty-mode "random" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dSCRencodedenv_2021_12_12__16_46_51_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###   --n-episodes $1 --build-name "staticasl" --difficulty-mode "medium" --render False # oh no this is before fix!
### # other seeds
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dSCRencodedenv_2021_12_10__19_34_45_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###   --n-episodes $1 --build-name "./alternate.x86_64" --difficulty-mode "hardest" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dSCRencodedenv_2021_12_10__19_34_45_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###   --n-episodes $1 --build-name "./city.x86_64" --difficulty-mode "hardest" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dSCRencodedenv_2021_12_10__19_34_45_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###   --n-episodes $1 --build-name "./office.x86_64" --difficulty-mode "random" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dSCRencodedenv_2021_12_10__19_34_45_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###   --n-episodes $1 --build-name "staticasl" --difficulty-mode "medium" --render False # oh no this is before fix!
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dSCRencodedenv_2021_12_06__21_58_01_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###   --n-episodes $1 --build-name "./alternate.x86_64" --difficulty-mode "hardest" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dSCRencodedenv_2021_12_06__21_58_01_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###   --n-episodes $1 --build-name "./city.x86_64" --difficulty-mode "hardest" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dSCRencodedenv_2021_12_06__21_58_01_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###   --n-episodes $1 --build-name "./office.x86_64" --difficulty-mode "random" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dSCRencodedenv_2021_12_06__21_58_01_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###   --n-episodes $1 --build-name "staticasl" --difficulty-mode "medium" --render False # oh no this is before fix!


# 6) part II, rest of the seeds
# not all models are trained fully!
python test_any.py --model-path \
  "~/navdreams_data/results/models/gym/navrep3dcathedralencodedenv_2022_02_14__10_22_45_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
  --n-episodes $1 --build-name "cathedral" --difficulty-mode "medium" --render False
python test_any.py --model-path \
  "~/navdreams_data/results/models/gym/navrep3dcathedralencodedenv_2022_02_14__10_23_10_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
  --n-episodes $1 --build-name "cathedral" --difficulty-mode "medium" --render False
python test_any.py --model-path \
  "~/navdreams_data/results/models/gym/navrep3dcathedralencodedenv_2022_02_16__15_07_50_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
  --n-episodes $1 --build-name "cathedral" --difficulty-mode "medium" --render False
python test_any.py --model-path \
  "~/navdreams_data/results/models/gym/navrep3dcathedralenv_2022_02_11__18_09_16_DISCRETE_PPO_E2E_VCARCH_C64_bestckpt.zip" \
  --n-episodes $1 --build-name "cathedral" --difficulty-mode "medium" --render False
python test_any.py --model-path \
  "~/navdreams_data/results/models/gym/navrep3dcathedralenv_2022_02_16__15_21_00_DISCRETE_PPO_E2E_VCARCH_C64_bestckpt.zip" \
  --n-episodes $1 --build-name "cathedral" --difficulty-mode "medium" --render False
python test_any.py --model-path \
  "~/navdreams_data/results/models/gym/navrep3dcathedralenv_2022_02_17__15_24_20_DISCRETE_PPO_E2E_VCARCH_C64_bestckpt.zip" \
  --n-episodes $1 --build-name "cathedral" --difficulty-mode "medium" --render False

python test_any.py --model-path \
  "~/navdreams_data/results/models/gym/navrep3dgalleryencodedenv_2022_02_11__21_52_34_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
  --n-episodes $1 --build-name "gallery" --difficulty-mode "easy" --render False
python test_any.py --model-path \
  "~/navdreams_data/results/models/gym/navrep3dgalleryencodedenv_2022_02_11__21_55_17_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
  --n-episodes $1 --build-name "gallery" --difficulty-mode "easy" --render False
python test_any.py --model-path \
  "~/navdreams_data/results/models/gym/navrep3dgalleryencodedenv_2022_02_16__15_23_37_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
  --n-episodes $1 --build-name "gallery" --difficulty-mode "easy" --render False
python test_any.py --model-path \
  "~/navdreams_data/results/models/gym/navrep3dgalleryenv_2022_02_16__15_08_38_DISCRETE_PPO_E2E_VCARCH_C64_bestckpt.zip" \
  --n-episodes $1 --build-name "gallery" --difficulty-mode "easy" --render False
python test_any.py --model-path \
  "~/navdreams_data/results/models/gym/navrep3dgalleryenv_2022_02_16__15_19_22_DISCRETE_PPO_E2E_VCARCH_C64_bestckpt.zip" \
  --n-episodes $1 --build-name "gallery" --difficulty-mode "easy" --render False
python test_any.py --model-path \
  "~/navdreams_data/results/models/gym/navrep3dgalleryenv_2022_02_17__15_23_22_DISCRETE_PPO_E2E_VCARCH_C64_bestckpt.zip" \
  --n-episodes $1 --build-name "gallery" --difficulty-mode "easy" --render False

python test_any.py --model-path \
  "~/navdreams_data/results/models/gym/navrep3dkozehdrsencodedenv_2022_02_02__17_18_59_DISCRETE_PPO_GPT_V_ONLY_V64M64_K2_bestckpt.zip" \
  --n-episodes $1 --build-name "kozehd" --difficulty-mode "easier" --render False
python test_any.py --model-path \
  "~/navdreams_data/results/models/gym/navrep3dkozehdrsencodedenv_2022_02_06__22_45_40_DISCRETE_PPO_GPT_V_ONLY_V64M64_K2_bestckpt.zip" \
  --n-episodes $1 --build-name "kozehd" --difficulty-mode "easier" --render False
python test_any.py --model-path \
  "~/navdreams_data/results/models/gym/navrep3dkozehdrsencodedenv_2022_02_22__10_42_42_DISCRETE_PPO_GPT_V_ONLY_V64M64_K2_bestckpt.zip" \
  --n-episodes $1 --build-name "kozehd" --difficulty-mode "easier" --render False
python test_any.py --model-path \
  "~/navdreams_data/results/models/gym/navrep3dkozehdrsencodedenv_2022_02_06__22_58_00_DISCRETE_PPO_E2E_VCARCH_C64_bestckpt.zip" \
  --n-episodes $1 --build-name "kozehd" --difficulty-mode "easier" --render False
python test_any.py --model-path \
  "~/navdreams_data/results/models/gym/navrep3dkozehdrsencodedenv_2022_02_09__18_19_21_DISCRETE_PPO_E2E_VCARCH_C64_bestckpt.zip" \
  --n-episodes $1 --build-name "kozehd" --difficulty-mode "easier" --render False
python test_any.py --model-path \
  "~/navdreams_data/results/models/gym/navrep3dkozehdrsenv_2022_02_23__10_41_13_DISCRETE_PPO_E2E_VCARCH_C64_bestckpt.zip" \
  --n-episodes $1 --build-name "kozehd" --difficulty-mode "easier" --render False


# Redo kozehd with 3 people
# python test_any.py --model-path \
#   "~/navdreams_data/results/models/gym/navrep3dkozehdrsencodedenv_2022_02_02__17_18_59_DISCRETE_PPO_GPT_V_ONLY_V64M64_K2_bestckpt.zip" \
#   --n-episodes $1 --build-name "kozehd" --difficulty-mode "easier" --render False
### cd ~/Code/pydreamer
### python test.py --run-id "3aaa8d09bce64dd888240a04b714aec7" --n-episodes $1 --build-name "kozehd" --difficulty-mode "easier" --render False
### cd ~/Code/cbsim/navrep3d
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dkozehdrsencodedenv_2022_02_06__22_58_00_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" \
###   --n-episodes $1 --build-name "kozehd" --difficulty-mode "easier" --render False
### # S, SC, SCR controllers
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3daltencodedenv_2021_12_15__08_43_12_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###   --n-episodes $1 --build-name "kozehd" --difficulty-mode "easier" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dSCencodedenv_2021_12_10__19_33_51_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###   --n-episodes $1 --build-name "kozehd" --difficulty-mode "easier" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dSCRencodedenv_2021_12_12__16_46_51_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
###   --n-episodes $1 --build-name "kozehd" --difficulty-mode "easier" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3daltenv_2021_11_15__16_16_40_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" \
###   --n-episodes $1 --build-name "kozehd" --difficulty-mode "easier" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dSCenv_2021_12_16__01_55_07_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" \
###   --n-episodes $1 --build-name "kozehd" --difficulty-mode "easier" --render False
### python test_any.py --model-path \
###   "~/navdreams_data/results/models/gym/navrep3dSCRencodedenv_2021_12_05__13_26_46_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" \
###   --n-episodes $1 --build-name "kozehd" --difficulty-mode "easier" --render False



# DREAMERV2
#         run_id = "f3f47a18b9334a4baa97c728143a00c6" # "./alternate.x86_64"
#         run_id = "0657e4d7a0f14c6ea301017f6774402b" # "./alternate.x86_64"
#         run_id = "a1ec5269279f46f79af2884526590592" # "staticasl" (staticaslfixed)
#         run_id = "3aaa8d09bce64dd888240a04b714aec7" # "kozehd" (kozehdrs)
# ------------------------------------------------------------------------

# cd ~/Code/pydreamer
# python test.py --run-id "f3f47a18b9334a4baa97c728143a00c6" --n-episodes $1 --build-name "./alternate.x86_64" --difficulty-mode "hardest" --render False
# python test.py --run-id "0657e4d7a0f14c6ea301017f6774402b" --n-episodes $1 --build-name "./alternate.x86_64" --difficulty-mode "hardest" --render False
# python test.py --run-id "a1ec5269279f46f79af2884526590592" --n-episodes $1 --build-name "staticasl" --difficulty-mode "hardest" --render False
# python test.py --run-id "3aaa8d09bce64dd888240a04b714aec7" --n-episodes $1 --build-name "kozehd" --difficulty-mode "hardest" --render False

