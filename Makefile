clean:
	docker rm -f $$(docker ps -qa)

build-image:
	docker build -t=deeppreid:v0 .

run:
	nvidia-docker run -v ${PWD}:/home/appuser --name=deeppreid --net=host --ipc=host -it deeppreid:v0
