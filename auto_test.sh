set -x
set -e
source ~/cbc3env/bin/activate

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
# 2) which models do best on gallery, cathedral, kozehdrs?
# 3) does a R model trained on SC do well in R?
# 4) does scope analysis also hold up at test_time?

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
#     "~/navrep3d/models/gym/navrep3daltenv_2021_11_01__08_52_03_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip"
    # n3d
#     "~/navrep3d/models/gym/navrep3daslencodedenv_2021_12_11__00_23_55_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip"
#     "~/navrep3d/models/gym/navrep3daslencodedenv_2021_12_08__10_18_09_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip"
#     "~/navrep3d/models/gym/navrep3dSCRencodedenv_2021_12_12__16_46_51_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip"
#     "~/navrep3d/models/gym/navrep3dkozehdencodedenv_2022_01_17__12_55_53_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip"
#     "~/navrep3d/models/gym/navrep3dkozehdencodedenv_2022_01_17__12_55_53_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip"
#     "~/navrep3d/models/gym/navrep3dkozehdoldencodedenv_2022_01_13__16_13_45_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip"
#     "~/navrep3d/models/gym/navrep3dkozehdoldencodedenv_2022_01_13__16_13_45_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip"
#     "~/navrep3d/models/gym/navrep3dkozehdremptyencodedenv_2022_01_30__22_12_30_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCRK_bestckpt.zip"
#     "~/navrep3d/models/gym/navrep3dkozehdremptyencodedenv_2022_01_30__22_12_30_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCRK_ckpt.zip"
#     "~/navrep3d/models/gym/navrep3daltencodedenv_2021_12_15__08_43_12_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip"
#     "~/navrep3d/models/gym/navrep3daslfixedencodedenv_2021_12_29__17_17_16_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip"
    # untested
#     "~/navrep3d/models/gym/navrep3dkozehdrencodedenv_2022_01_28__06_44_50_DISCRETE_PPO_GPT_V_ONLY_V64M64_K2_ckpt.zip"
#     "~/navrep3d/models/gym/navrep3dkozehdrencodedenv_2022_01_25__21_34_44_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCRK_ckpt.zip"
#     "~/navrep3d/models/gym/navrep3dkozehdrencodedenv_2022_01_26__08_56_54_DISCRETE_PPO_GPT_V_ONLY_V64M64_K_ckpt.zip"
#     "~/navrep3d/models/gym/navrep3dkozehdrsencodedenv_2022_02_02__17_18_59_DISCRETE_PPO_GPT_V_ONLY_V64M64_K2_bestckpt.zip"
# ------------------------------------------------------------------------

cd ~/Code/cbsim/navrep3d
# 2)
python test_any.py --model-path \
  "~/navrep3d/models/gym/navrep3daltencodedenv_2021_12_15__08_43_12_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
  --n-episodes 100 --build-name "gallery" --difficulty-mode "easy" --render False
python test_any.py --model-path \
  "~/navrep3d/models/gym/navrep3dSCencodedenv_2021_12_10__19_33_51_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
  --n-episodes 100 --build-name "gallery" --difficulty-mode "easy" --render False
python test_any.py --model-path \
  "~/navrep3d/models/gym/navrep3dSCRencodedenv_2021_12_12__16_46_51_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
  --n-episodes 100 --build-name "gallery" --difficulty-mode "easy" --render False
python test_any.py --model-path \
  "~/navrep3d/models/gym/navrep3daltenv_2021_11_15__16_16_40_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" \
  --n-episodes 100 --build-name "gallery" --difficulty-mode "easy" --render False
python test_any.py --model-path \
  "~/navrep3d/models/gym/navrep3dSCenv_2021_12_16__01_55_07_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" \
  --n-episodes 100 --build-name "gallery" --difficulty-mode "easy" --render False
python test_any.py --model-path \
  "~/navrep3d/models/gym/navrep3dSCRencodedenv_2021_12_05__13_26_46_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" \
  --n-episodes 100 --build-name "gallery" --difficulty-mode "easy" --render False
# cathedral
python test_any.py --model-path \
  "~/navrep3d/models/gym/navrep3daltencodedenv_2021_12_15__08_43_12_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
  --n-episodes 100 --build-name "cathedral" --difficulty-mode "easy" --render False
python test_any.py --model-path \
  "~/navrep3d/models/gym/navrep3dSCencodedenv_2021_12_10__19_33_51_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
  --n-episodes 100 --build-name "cathedral" --difficulty-mode "easy" --render False
python test_any.py --model-path \
  "~/navrep3d/models/gym/navrep3dSCRencodedenv_2021_12_12__16_46_51_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
  --n-episodes 100 --build-name "cathedral" --difficulty-mode "easy" --render False
python test_any.py --model-path \
  "~/navrep3d/models/gym/navrep3daltenv_2021_11_15__16_16_40_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" \
  --n-episodes 100 --build-name "cathedral" --difficulty-mode "easy" --render False
python test_any.py --model-path \
  "~/navrep3d/models/gym/navrep3dSCenv_2021_12_16__01_55_07_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" \
  --n-episodes 100 --build-name "cathedral" --difficulty-mode "easy" --render False
python test_any.py --model-path \
  "~/navrep3d/models/gym/navrep3dSCRencodedenv_2021_12_05__13_26_46_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" \
  --n-episodes 100 --build-name "cathedral" --difficulty-mode "easy" --render False
# kozehd
python test_any.py --model-path \
  "~/navrep3d/models/gym/navrep3daltencodedenv_2021_12_15__08_43_12_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
  --n-episodes 100 --build-name "cathedral" --difficulty-mode "easiest" --render False
python test_any.py --model-path \
  "~/navrep3d/models/gym/navrep3dSCencodedenv_2021_12_10__19_33_51_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
  --n-episodes 100 --build-name "cathedral" --difficulty-mode "easiest" --render False
python test_any.py --model-path \
  "~/navrep3d/models/gym/navrep3dSCRencodedenv_2021_12_12__16_46_51_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
  --n-episodes 100 --build-name "cathedral" --difficulty-mode "easiest" --render False
python test_any.py --model-path \
  "~/navrep3d/models/gym/navrep3daltenv_2021_11_15__16_16_40_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" \
  --n-episodes 100 --build-name "cathedral" --difficulty-mode "easiest" --render False
python test_any.py --model-path \
  "~/navrep3d/models/gym/navrep3dSCenv_2021_12_16__01_55_07_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" \
  --n-episodes 100 --build-name "cathedral" --difficulty-mode "easiest" --render False
python test_any.py --model-path \
  "~/navrep3d/models/gym/navrep3dSCRencodedenv_2021_12_05__13_26_46_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" \
  --n-episodes 100 --build-name "cathedral" --difficulty-mode "easiest" --render False
# 1)
python test_any.py --model-path \
  "~/navrep3d/models/gym/navrep3daltencodedenv_2021_12_15__08_43_12_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_ckpt.zip" \
  --n-episodes 100 --build-name "./alternate.x86_64" --difficulty-mode "hardest" --render False
python test_any.py --model-path \
  "~/navrep3d/models/gym/navrep3daltenv_2021_11_01__08_52_03_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip" \
  --n-episodes 100 --build-name "./alternate.x86_64" --difficulty-mode "hardest" --render False

# DREAMERV2
#         run_id = "f3f47a18b9334a4baa97c728143a00c6" # "./alternate.x86_64"
#         run_id = "0657e4d7a0f14c6ea301017f6774402b" # "./alternate.x86_64"
#         run_id = "a1ec5269279f46f79af2884526590592" # "staticasl" (staticaslfixed)
#         run_id = "3aaa8d09bce64dd888240a04b714aec7" # "kozehd" (kozehdrs)
# ------------------------------------------------------------------------

cd ~/Code/pydreamer
python test.py --run-id "f3f47a18b9334a4baa97c728143a00c6" --n-episodes 100 --build-name "./alternate.x86_64" --difficulty-mode "hardest" --render False
# python test.py --run-id "0657e4d7a0f14c6ea301017f6774402b" --n-episodes 100 --build-name "./alternate.x86_64" --difficulty-mode "hardest" --render False
# python test.py --run-id "a1ec5269279f46f79af2884526590592" --n-episodes 100 --build-name "staticasl" --difficulty-mode "hardest" --render False
# python test.py --run-id "3aaa8d09bce64dd888240a04b714aec7" --n-episodes 100 --build-name "kozehd" --difficulty-mode "hardest" --render False

