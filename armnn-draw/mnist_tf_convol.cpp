//
// Copyright © 2018 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <memory>
#include <array>
#include <algorithm>
#include <cstring>
#include "armnn/ArmNN.hpp"
#include "armnn/Exceptions.hpp"
#include "armnn/Tensor.hpp"
#include "armnn/INetwork.hpp"
#include "armnnTfParser/ITfParser.hpp"

#include "mnist_loader.hpp"


// Helper function to make input tensors
armnn::InputTensors MakeInputTensors(const std::pair<armnn::LayerBindingId,
    armnn::TensorInfo>& input,
    const void* inputTensorData)
{
    return { { input.first, armnn::ConstTensor(input.second, inputTensorData) } };
}

// Helper function to make output tensors
armnn::OutputTensors MakeOutputTensors(const std::pair<armnn::LayerBindingId,
    armnn::TensorInfo>& output,
    void* outputTensorData)
{
    return { { output.first, armnn::Tensor(output.second, outputTensorData) } };
}

int main(int argc, char** argv)
{
    // Default input size is one image
    unsigned int nrOfImages = 1;

    // Optimisation mode 0 = CPU Reference (unoptimised)
    // Optimisation mode 1 = CPU Accelerator
    // Optimisation mode 2 = GPU Accelerator
    unsigned int optimisationMode = 1;

    // Check program arguments
    if (argc != 3)
    {
      std::cerr << "Invalid arguments. Exactly 2 arguments are required." << std::endl;
      std::cerr << "Example: ./mnist_tf 1 image.txt" << std::endl;
      std::cerr << "Optimisation modes: 0 for CpuRef, 1 for CpuAcc, 2 for GpuAcc" << std::endl;
      std::cerr << "Input file: path to simple text file with mnist image pixel values raning from 0 to 255" << std::endl;
      return 1;
    }

    // Parse optimisation mode option
    optimisationMode = std::stoi(argv[1]);
    if (!(optimisationMode == 0 || optimisationMode == 1 || optimisationMode == 2))
    {
      std::cerr << "Invalid optimisation mode." << std::endl;
      return 1;
    }

    // Import the TensorFlow model. Note: use CreateNetworkFromBinaryFile for .pb files.
    armnnTfParser::ITfParserPtr parser = armnnTfParser::ITfParser::Create();
    armnn::INetworkPtr network = parser->CreateNetworkFromBinaryFile("model/optimized_mnist_tf.pb",
                                                                   { {"input_tensor", {nrOfImages, 784, 1, 1}} },
                                                                   { "fc2/output_tensor" });

    // Find the binding points for the input and output nodes
    armnnTfParser::BindingPointInfo inputBindingInfo = parser->GetNetworkInputBindingInfo("input_tensor");
    armnnTfParser::BindingPointInfo outputBindingInfo = parser->GetNetworkOutputBindingInfo("fc2/output_tensor");

    // Create ArmNN runtime
    armnn::IRuntime::CreationOptions options; // default options
    armnn::IRuntimePtr runtime = armnn::IRuntime::Create(options);

    // Optimize the network for a specific runtime compute device, e.g. CpuAcc, GpuAcc
    armnn::Compute device;
    switch(optimisationMode)
    {
      case 0: device = armnn::Compute::CpuRef; break;
      case 1: device = armnn::Compute::CpuAcc; break;
      case 2: device = armnn::Compute::GpuAcc; break;
    }

    armnn::IOptimizedNetworkPtr optNet = Optimize(*network, {device}, runtime->GetDeviceSpec());

    // Load the optimized network onto the runtime device
    armnn::NetworkId networkIdentifier;
    runtime->LoadNetwork(networkIdentifier, std::move(optNet));

    // Load image from a text file
    std::string dataFile = argv[2];

    float input[g_kMnistImageByteSize];
    float output[10];
    int labels;

    std::unique_ptr<MnistImage> imgInfo = loadMnistImage(dataFile);
    if (imgInfo == nullptr)
        return 1;

    std::memcpy(input, imgInfo->image, sizeof(imgInfo->image));
    labels = imgInfo->label;

    // Execute network
    armnn::InputTensors inputTensor = MakeInputTensors(inputBindingInfo, &input[0]);
    armnn::OutputTensors outputTensor = MakeOutputTensors(outputBindingInfo, &output[0]);

    armnn::Status ret = runtime->EnqueueWorkload(networkIdentifier, inputTensor, outputTensor);

    // Write output to stderr
    float max = output[0];
    int label = 0;

    for (int j = 0; j < 10; ++j)
    {
      std::cerr << output[j] << " ";
      if (output[j] > max)
      {
         max = output[j];
         label = j;
      }

    }

    std::cerr << std::endl;

    return 0;
}
