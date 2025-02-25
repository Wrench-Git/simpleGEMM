NAME := gputiled
CX_ROOT := ./include


# Target rules
all: build

build: $(NAME)

$(NAME).o: $(NAME).cu
	nvcc -ccbin cl -I $(CX_ROOT)/ -m64  --threads 0 --use_fast_math -gencode arch=compute_89,code=sm_89 -o $(NAME).o -c $(NAME).cu
    
$(NAME): $(NAME).o   
	nvcc -ccbin cl -m64 -gencode arch=compute_89,code=sm_89 -o $(NAME) $(NAME).o

run: build
	$(EXEC) ./$(NAME)

testrun: build

clean:
	rm -f    $(NAME) $(NAME).o
	rm -rf   $(NAME)
echo:
	$(info NAME is $(NAME))
#
