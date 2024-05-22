
#!/bin/bash
docker container run --runtime=nvidia -v ${PWD}:/home/user/outpace_diffusion --name outpace_diffusion -it --rm erditmp/outpace:latest