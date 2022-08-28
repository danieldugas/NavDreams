
docker run -v $(pwd)/gibson_challenge_data_2021:/opt/iGibson/gibson2/data -v ~/navdreams:~/navdreams -v $(pwd)/simple_agent.py:/simple_agent.py -v $(pwd)/do.sh:/do.sh --runtime=nvidia --name=igibson -it my_submission /bin/bash

