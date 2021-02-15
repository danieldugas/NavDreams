Hi Fabien,

Here's an updated version where the DWA planner runs faster (5~10hz on my end)

https://drive.google.com/drive/folders/13sLUF90iyhwmBK6QAr5qBvZibjtUntc4?usp=sharing

I've also made a install_dependencies.sh script which pulls and pip installs the two required dependencies to a virtualenv

so on my end, things work if I:

(inside CB_challenge executable folder)
launch CB_challenge_...

(inside crowdbotsimcontrol folder)
./install_dependencies.sh
source ~/cbcenv/bin/activate
python core.py

Cheers,
Daniel
