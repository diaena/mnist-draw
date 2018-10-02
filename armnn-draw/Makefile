ARMNN_LIB = ${HOME}/armnn-devenv/armnn/build
ARMNN_INC = ${HOME}/armnn-devenv/armnn/include

all: convol

convol: mnist_tf_convol.cpp mnist_loader.hpp
		g++ -O3 -std=c++17 -I${HOME}/armnn-devenv/boost/install/include -I$(ARMNN_INC) mnist_tf_convol.cpp -o mnist_tf_convol -L$(ARMNN_LIB) -larmnn -larmnnTfParser -lpthread

clean:
	-rm -f mnist_tf_convol

