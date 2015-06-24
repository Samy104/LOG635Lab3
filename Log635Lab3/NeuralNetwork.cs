using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Xml.Serialization;

namespace Neural
{
    public class NeuralNetwork
    {
        public const double AcceptableError = 0.005;

        [XmlElement]
        public readonly int MaxDepth = 2000;

        [XmlIgnore]
        public double LearningRate { get; private set; }

        public readonly double BaseLearningRate;

        public delegate bool ValidAnswer(double[] nn, double[] real);
        public static Random Rand = new Random();

        public struct NetworkParameters
        {
            public int Inputs;
            public int Outputs;
            public ValidAnswer Checker;
            public double TrainingRate;
            public int Layers;
            public int NeuronsPerLayers;
            public int MaxDepth;
        }

        public struct TrainingResults
        {
            /// <summary>
            /// In order: Depth, Success and MSE
            /// </summary>
            public Tuple<int, int, double> BestIteration;
            public int success;
            public int count;
            public double mse;
            public TimeSpan duration;
        }

        [XmlElement]
        public List<NeuralLayer> Layers
        {
            get; set;
        }

        [XmlElement]
        public int NumInputs
        {
            get; set;
        }

        [XmlElement]
        public int NumOutputs { get; set; }

        [XmlIgnore]
        public ValidAnswer Checker { get; set; }

        public NeuralNetwork()
        { }

        public NeuralNetwork(NetworkParameters param)
            :this(param.Inputs, param.Outputs, param.Layers, param.NeuronsPerLayers, param.Checker)
        {
            MaxDepth = param.MaxDepth;
            BaseLearningRate = param.TrainingRate;
        }

        public NeuralNetwork(int inputs, int outputs, int numLayers, int neurons, ValidAnswer checker)
        {
            Checker = checker;
            NumInputs = inputs;
            NumOutputs = outputs;
            BaseLearningRate = 0.002;
            Layers = new List<NeuralLayer>(numLayers);

            int lastNeurons = inputs;
            Layers.Add(new InputNeuralLayer(inputs, this));
            for(int i = 1; i < numLayers-1; i++)
            {
                Layers.Add(new NeuralLayer(neurons, lastNeurons, this));
                lastNeurons = neurons;
            }
            Layers.Add(new OutputNeuralLayer(outputs, lastNeurons, this));
        }

        public double[] Forward(double[] input)
        {
            //Console.WriteLine("== Pushing data {0}", string.Join("=", input));
            var ret = ForwardImpl(input);
            //Console.WriteLine("   Received data {0}", string.Join("=", ret));
            return ret;
        }

        private double[] ForwardImpl(double[] input)
        {
            var output = input;
            foreach (var l in Layers)
            {
                output = l.Forward(output);
            }
            return output;
        }
        
        public TrainingResults TrainBatch(IEnumerable<double[]> batch, IEnumerable<double[]> result, Action<string> logger = null)
        {
            bool interrupted = false;
            if (logger == null)
                logger = s => Console.WriteLine(s);
            double mse = 0;
            int depth = 0;
            int correct = 0;
            int count = 0;
            int timesStagnant = 0;
            int lastCorrect = 0;
            double[] output;
            DateTime startTime = DateTime.UtcNow;
            Tuple<int, int, double> maxSuccess = Tuple.Create(0,0,0.0);
            var batchData = batch.Tuplify(result).ToList();
            LearningRate = BaseLearningRate;
            do
            {
                count = 0;
                correct = 0;
                // Do batch
                foreach (var data in batchData)
                {
                    output = ForwardImpl(data.Item1);
                    double[] expected = data.Item2;
                    for (int i = Layers.Count - 1; i >= 0; i--)
                    {
                        expected = Layers[i].ComputeGradients(expected);
                    }

                    for (int i = 0; i < output.Length; i++)
                    {
                        mse += Math.Pow(output[i] - data.Item2[i], 2);
                    }
                    if (Checker(output, data.Item2))
                        correct++;
                    count++;
                    if (count == 1)
                    {
                        //Console.WriteLine(string.Format("- {0,3} Output data : ", count) + string.Join("=", output.Select(v => string.Format("{0:#.00000000}",v))));
                        //Console.WriteLine("    Expected data : " + string.Join("=", data.Item2.Select(v => string.Format("{0:#.00000000}", v))));
                    }
                    //Console.ReadLine();
                }

                Layers.ForEach(l => l.UpdateWeights());
                
                mse = mse / (count * NumOutputs);
                depth++;
                // Boost the learning rate when we are not moving
                if (lastCorrect >= correct)
                    timesStagnant++;
                else
                {
                    lastCorrect = correct;
                    timesStagnant = 0;
                    LearningRate = BaseLearningRate < (LearningRate * 0.85) ? LearningRate * 0.85 : BaseLearningRate;
                }
                if (timesStagnant > 1 && timesStagnant % 50 == 1)
                {
                    LearningRate = LearningRate > BaseLearningRate*2.0 ? BaseLearningRate*2.0 : LearningRate * 1.5;
                    //Console.WriteLine("Boosting learning rate to " + LearningRate);
                }
                if (correct > maxSuccess.Item1)
                    maxSuccess = Tuple.Create(correct, depth, mse);
                if(depth%50 == 1)
                    logger(string.Format("Batch Iteration {0:0000}: MSE={1:00.00000000}% /\\ Success={2:#0.00}% ({3}/{4})", depth, mse * 100, ((double)correct) / count * 100, correct, count));
                //Console.ReadLine();
            } while (mse > AcceptableError && depth < MaxDepth);
            logger(string.Format("Batch Training Done in {0} : MSE={1}% /\\ Success={2}% ({3}/{4})", DateTime.UtcNow-startTime, mse * 100, ((double)correct) / count * 100, correct, count));
            //Console.WriteLine(string.Format("Best depth at {0}. Had {1} successes and {2} mse", maxSuccess.Item2, maxSuccess.Item1, maxSuccess.Item3 * 100));

            return new TrainingResults
            {
                BestIteration = maxSuccess,
                count = count,
                duration = DateTime.UtcNow-startTime,
                mse = mse,
                success = correct
            };
        }

        public void SetNetwork()
        {
            Layers.ForEach(l => l.SetNetwork(this));
        }
    }

    public class InputNeuralLayer : NeuralLayer
    {
        public InputNeuralLayer(int inputs, NeuralNetwork net)
            :base(0, inputs, net)
        {
        }

        public InputNeuralLayer() { }

        public override double[] Forward(double[] input)
        {
            //Console.WriteLine("--- Pushing data forward : " + string.Join("=", input));
            var output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i].Sigmoid();
            }
            return output;
        }

        public override double[] ComputeGradients(double[] expected)
        {
            return null;
        }

        public override void UpdateWeights()
        {
            return;
        }

        public override void SetNetwork(NeuralNetwork net)
        {
            return;
        }
    }

    public class OutputNeuralLayer : NeuralLayer
    {
        public OutputNeuralLayer(int outputs, int inputs, NeuralNetwork net)
            : base(outputs, inputs, net)
        {
            Neurons[0].Debug = false;
        }

        public OutputNeuralLayer() { }

        public override double[] Forward(double[] input)
        {
            var ret = base.Forward(input);
            //Console.WriteLine("--- Outputing data : " + string.Join("=", ret));
            return ret;
        }

        public override double[] ComputeGradients(double[] expected)
        {
            var expectedIn = new double[NumInputs];
            for (int i = 0; i < Neurons.Length; i++)
            {
                var wo = Neurons[i].ComputeGradient(expected[i] - Neurons[i].Output);
                for (int j = 0; j < NumInputs; j++)
                {
                    expectedIn[j] += wo[j];
                }
            }
            return expectedIn;
        }
    }

    [XmlInclude(typeof(InputNeuralLayer)), XmlInclude(typeof(OutputNeuralLayer))]
    public class NeuralLayer
    {
        protected double[] output;

        [XmlElement]
        public Neuron[] Neurons { get; set; }

        [XmlElement]
        public int NumInputs { get; set; }

        public virtual double[] Forward(double[] input)
        {
            output = new double[Neurons.Length];
            for(int i = 0; i < Neurons.Length; i++)
            {
                output[i] = Neurons[i].Forward(input);
            }
            return output;
        }

        public NeuralLayer(int numNeurons, int inputs, NeuralNetwork net)
        {
            Neurons = new Neuron[numNeurons];
            NumInputs = inputs;
            for(int i = 0; i < numNeurons; i++)
            {
                Neurons[i] = new Neuron(inputs, net);
            }
            //Neurons[0].Debug = true;
        }

        public NeuralLayer() { }

        public virtual void SetNetwork(NeuralNetwork net)
        {
            Neurons.ForEach(n => n.SetNetwork(net));
        }

        public virtual double[] ComputeGradients(double[] expected)
        {
            double[] expectedIn = null;
            for(int i = 0; i < Neurons.Length; i++)
            {
                var neuronExpected = Neurons[i].ComputeGradient(expected[i]);
                if (expectedIn == null)
                    expectedIn = neuronExpected;
                else
                    for (int j = 0; j < NumInputs; j++)
                        expectedIn[j] += neuronExpected[j];
            }
            return expectedIn;
        }

        public virtual void UpdateWeights()
        {
            Neurons.ForEach(n => n.UpdateWeights());
        }
    }

    public class Neuron
    {
        private double output;
        private double[] inputs;
        private double[] deltas;
        private double deltaBias = 0;

        [XmlElement]
        public double[] Weights { get; set; }

        [XmlElement]
        public double Bias { get; set; }

        public bool Debug
        {
            get; set;
        }

        public double Output
        {
            get
            {
                return output;
            }
        }
        
        [XmlIgnore]
        public NeuralNetwork Network { get; private set; }

        public Neuron(int inputs, NeuralNetwork net) 
        {
            Network = net;
            Weights = new double[inputs];
            deltas = new double[inputs];
            for (int i = 0; i < inputs; i++)
            {
                Weights[i] = NeuralNetwork.Rand.NextDouble()*4-2;
            }
            Bias = NeuralNetwork.Rand.NextDouble()*4-2;
        }

        public Neuron()
        {
        }

        public void SetNetwork(NeuralNetwork net)
        {
            Network = net;
        }

        public double Forward(double[] inputs)
        {
            this.inputs = inputs;
            if (inputs.Length != Weights.Length)
            {
                throw new InvalidOperationException();
            }
            output = 0;
            for (int i = 0; i < Weights.Length; i++)
            {
                output += inputs[i] * Weights[i];
            }
            output = (output + Bias).Sigmoid();
            return output;
        }
        
        public double[] ComputeGradient(double error)
        {
            var gradient = output * (1 - output) * error;
            for (int i = 0; i < deltas.Length; i++)
            {
                deltas[i] += Network.LearningRate * inputs[i] * gradient;
            }
            deltaBias += Network.LearningRate * gradient;
            if (Debug)
            {
                Console.WriteLine(string.Format("--- Gradient computed to {0}. Error was {1}. Delta Bias is {2}", gradient, error, deltaBias));
            }
            return Weights.Select(w => w * gradient).ToArray();
        }
        
        public void UpdateWeights()
        {
            if (Debug)
            {
                Console.WriteLine(string.Format("--- Weights before adjust : {0}", string.Join("=", Weights)));
            }
            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] += deltas[i];
            }
            Bias += deltaBias;
            if (Debug)
            {
                Console.WriteLine(string.Format("    Weights after adjust :  {0}", string.Join("=", Weights)));
                Console.WriteLine(string.Format("    Deltas : {0}", string.Join("=", deltas)));
                Console.WriteLine(string.Format("    DeltaBias : {0}", deltaBias));
                Console.WriteLine(string.Format("    Input : {0}", string.Join("=", inputs)));
            }
            deltaBias = 0;
            Array.Clear(deltas, 0, deltas.Length);
        }
    }
}
