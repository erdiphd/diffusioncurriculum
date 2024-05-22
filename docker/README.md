# Dockerfile for Outpace


## To create docker image 
# (To be able to copy the repository this line in the Dockerfile COPY outpace_official /home/user/outpace_official), you need to build the docker container with follwing command
`docker build -t erditmp/outpace_diffusion:latest -f Dockerfile .`

## To run HGG docker container

`docker container run --runtime=nvidia -v ${PWD}:/home/user/diffusion_curriculum --name diffusion_curriculum -it --rm erditmp/outpace`




