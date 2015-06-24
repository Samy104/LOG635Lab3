using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;
using Neural;

namespace Log635Lab3
{
    public enum League
    {
        Unknown = 0,
        Bronze = 1,
        Silver = 2,
        Or = 3,
        Platinum = 4,
        Diamant = 5,
        Maitre = 6,
        GrandMaitre = 7,
        Professionnel = 8,

        Max = 8
    }

    public class Player
    {
        private double[] _input;

        public int GameId { get; set; }
        public League League { get; set; }
        public int Age { get; set; }
        public int HoursPerWeek { get; set; }
        public int TotalHours { get; set; }
        public double APM { get; set; }
        public double SelectByHotkey { get; set; }
        public double AssignToHotkey { get; set; }
        public double UniqueHotkey { get; set; }
        public double MinimapAttacks { get; set; }
        public double MinimapRightClicks { get; set; }
        public double NumberOfPacs { get; set; }
        public double GapBetweenPacs { get; set; }
        public double ActionLatency { get; set; }
        public double ActionsInPacs { get; set; }
        public double TotalMapExplored { get; set; }
        public double WorkersMade { get; set; }
        public double UniqueUnitsMade { get; set; }
        public double ComplexUnitMade { get; set; }
        public double ComplexAbilitiesUsed { get; set; }

        public override string ToString()
        {
            var output = new StringBuilder();
            output.AppendFormat("--- Game {0} for league {1}\n", GameId, League.ToString());
            output.AppendFormat("\tAge : {0} // HPW : {1} // TotalH : {2} // APM : {3}\n", Age, HoursPerWeek, TotalHours, APM);
            output.AppendFormat("\tSBH : {0} // ATH : {1} // UHotKey : {2} // MmapAtk : {3}\n", SelectByHotkey, AssignToHotkey, UniqueHotkey, MinimapAttacks);
            output.AppendFormat("\tMmapRC : {0} // NumPACS : {1} // GapPACS : {2} // ActLatency : {3}\n", MinimapRightClicks, NumberOfPacs, GapBetweenPacs, ActionLatency);
            output.AppendFormat("\tActPacs : {0} // TotMap : {1} // Workers : {2}\n", ActionsInPacs, TotalMapExplored, WorkersMade);
            output.AppendFormat("\tUniqUnits : {0} // ComplexUnits : {1} // ComplexAbilities : {2}", UniqueUnitsMade, ComplexUnitMade, ComplexAbilitiesUsed);
            return output.ToString();
        }

        public double[] GetInput()
        {
            if (_input != null)
            {
                return _input;
            }
            _input = new double[18];
            _input[0]  = Age;
            _input[1]  = HoursPerWeek;
            _input[2]  = TotalHours;
            _input[3]  = APM;
            _input[4]  = SelectByHotkey;
            _input[5]  = AssignToHotkey;
            _input[6]  = UniqueHotkey;
            _input[7]  = MinimapAttacks;
            _input[8]  = MinimapRightClicks;
            _input[9]  = NumberOfPacs;
            _input[10] = GapBetweenPacs;
            _input[11] = ActionLatency;
            _input[12] = ActionsInPacs;
            _input[13] = TotalMapExplored;
            _input[14] = WorkersMade;
            _input[15] = UniqueUnitsMade;
            _input[16] = ComplexUnitMade;
            _input[17] = ComplexAbilitiesUsed;
            return _input;
        }
    }

    public class SerializeNetwork
    {
        [XmlElement] public double[] means;
        [XmlElement] public double[] stdev;
        [XmlElement] public NeuralNetwork network;
    }

    class Program
    {
        public static NeuralNetwork.ValidAnswer checker = (artificial, real) =>
        {
            return artificial.ToLeague() == real.ToLeague();
        };

        private static NeuralNetwork network;
        private static CultureInfo culture = CultureInfo.CreateSpecificCulture("en-US");
        private static double[] mean;
        private static double[] stdev;

        static void Main(string[] args)
        {
            if (args.Length < 2)
            {

                Environment.Exit(1);
                return;
            }

            
            var test = extractPlayers(args[1]);

            //Console.Out.WriteLine(allPlayers.Count);
            var serializer = new XmlSerializer(typeof(SerializeNetwork));
            if (args[0].EndsWith(".xml"))
            {
                // Training is a xml backup
                var load = File.OpenRead(args[0]);
                var deserialized = (SerializeNetwork)serializer.Deserialize(load);
                load.Close();
                network = deserialized.network;
                mean = deserialized.means;
                stdev = deserialized.stdev;
                network.Checker = checker;
                network.SetNetwork();
            }
            else
            {
                var training = extractPlayers(args[0]);
                network = Train(training);
            }
            Test(test);

            var save = File.Create("network.xml");
            serializer.Serialize(save, new SerializeNetwork { network = network, means = mean, stdev = stdev });
            save.Close();
            while(Console.ReadLine() != "stop");
        }
        
        private static NeuralNetwork Train(List<Player> dataSet)
        {
            mean = new double[18];
            var counts = new int[18];
            for (int i = 0; i < mean.Length; i++)
            {
                double sum = 0;
                foreach(var p in dataSet.Select(p => p.GetInput()))
                {
                    if (p[i] == -1 || p[i] > 20000)
                        continue;
                    sum += p[i];
                    counts[i]++;
                }
                mean[i] = sum / counts[i];
            }
            
            stdev= new double[18];
            for (int i = 0; i < mean.Length; i++)
            {
                double sum = 0;
                foreach(var p in dataSet.Select(p => p.GetInput()))
                {
                    if (p[i] == -1 || p[i] > 20000)
                        continue;
                    sum += Math.Pow(p[i]-mean[i], 2);
                }
                stdev[i] = Math.Sqrt(sum / counts[i]);
            }

            //Console.WriteLine(string.Join("//", allP[0].GetInput()) + "\n");
            //Console.WriteLine(string.Join("//", mean) + "\n");
            //Console.WriteLine(string.Join("//", stdev) + "\n");
            //Console.WriteLine(string.Join("//", allP[0].GetInput().Normalize(mean, stdev)));

            // TODO : Needs work
            //Debug.WriteLine(allP[0].League + " 1 : " + string.Join("-", allP[0].GetInput().Normalize(mean, stdev)));
            //Debug.WriteLine(allP[1].League + " 2 : " + string.Join("-", allP[1].GetInput().Normalize(mean, stdev)));
            //Debug.WriteLine(allP[2].League + " 2 : " + string.Join("-", allP[2].GetInput().Normalize(mean, stdev)));
            if (dataSet[0].League != dataSet[0].League.ToOutput().ToLeague())
            {
                Console.WriteLine("Retard dumbass: "+ dataSet[0].League + "--" + dataSet[0].League.ToOutput().ToLeague());
                Console.Read();
            }

            // Do training
            dataSet = dataSet.Shuffle(NeuralNetwork.Rand).ToList();
            //var results = network.TrainBatch(allP.Take(trainSet).Select(p => p.GetInput().Normalize(mean, stdev)).ToList(), allP.Take(trainSet).Select(p => p.League.ToOutput()));
            //Console.WriteLine(string.Format("Finished train in {0}ms", results.duration.TotalSeconds));

            var trainData = dataSet.Select(p => p.GetInput().Normalize(mean, stdev)).ToList();
            var trainResults = dataSet.Select(p => p.League.ToOutput()).ToList();
            var opti = new OptimizationTrier(Tuple.Create(trainData.AsEnumerable(), trainResults.AsEnumerable()));
            Console.WriteLine("Starting the beast");
            var network = opti.Run();
            
            Console.ReadLine();
            
            return network;
        }

        private static void Test(List<Player> dataSet)
        {
            // Check
            var count = 0;
            var success = 0;
            double error = 0;
            for (int i = 0; i < dataSet.Count; i++)
            {
                var output = network.Forward(dataSet[i].GetInput().Normalize(mean, stdev));
                var leagueOut = output.ToLeague();
                var leagueReal = dataSet[i].League;

                //Console.Out.WriteLine("Output is : " + nn.Output[0]);
                Console.WriteLine("Player {2} is {0}. Result is {1}", leagueReal, leagueOut, i);
                count++;
                error += Math.Abs((int)leagueReal - (int)leagueOut);
                if (checker(output, dataSet[i].League.ToOutput()))
                    success++;
            }
            error /= count;
            Console.WriteLine("Success of {0}% ({1}/{2}). Mean error = {3}", ((double)success / count) * 100, success, count, error);
        }

        private static List<Player> extractPlayers(string path)
        {
            var file = new StreamReader(path);
            file.ReadLine(); // Skip the titles line

            Func<string, string> sanitize = s =>
            {
                s = s.Replace("\\", "");
                s = s.Replace("\"", "");
                s = s.Replace("?", "-1");
                return s;
            };

            string line;
            var players = new List<Player>();
            while ((line = file.ReadLine()) != null)
            {
                var data = line.Split(',');
                Player p = new Player();
                p.GameId = int.Parse(data[0]);
                p.League = (data[1] == "?" ? League.Unknown : (League)int.Parse(data[1], culture));
                p.Age = int.Parse(sanitize(data[2]), culture);
                p.HoursPerWeek = int.Parse(sanitize(data[3]), culture);
                p.TotalHours = int.Parse(sanitize(data[4]), culture);
                p.APM = double.Parse(sanitize(data[5]), culture);
                p.SelectByHotkey = double.Parse(sanitize(data[6]), culture);
                p.AssignToHotkey = double.Parse(sanitize(data[7]), culture);
                p.UniqueHotkey = double.Parse(sanitize(data[8]), culture);
                p.MinimapAttacks = double.Parse(sanitize(data[9]), culture);
                p.MinimapRightClicks = double.Parse(sanitize(data[10]), culture);
                p.NumberOfPacs = double.Parse(sanitize(data[11]), culture);
                p.GapBetweenPacs = double.Parse(sanitize(data[12]), culture);
                p.ActionLatency = double.Parse(sanitize(data[13]), culture);
                p.ActionsInPacs = double.Parse(sanitize(data[14]), culture);
                p.TotalMapExplored = double.Parse(sanitize(data[15]), culture);
                p.WorkersMade = double.Parse(sanitize(data[16]), culture);
                p.UniqueUnitsMade = double.Parse(sanitize(data[17]), culture);
                p.ComplexUnitMade = double.Parse(sanitize(data[18]), culture);
                p.ComplexAbilitiesUsed = double.Parse(sanitize(data[19]), culture);

                players.Add(p);
            }

            return players;
        }
    }

    public static class StarCraftExtensions
    {
        public static double[] ToOutput(this League league)
        {
            //return ((double)league-4).Sigmoid().SimpleArray();
            
            const double basicActivation = 1;
            const double step = 2;

            var val = (int)league;
            var output = new double[(int)League.Max];
            //Console.WriteLine("----");
            for(int i = 1; i <= (int)League.Max; i++)
            {
                //Console.WriteLine((basicActivation + (val - i) * step));
                output[i-1] = (basicActivation + (val - i) * step).Sigmoid();
                //Console.WriteLine(output[i - 1]);
            }
            return output;
        }

        public static League ToLeague(this double[] list)
        {
            //return (League)Math.Round(list[0].InvSigmoid()+4);
            
            if (list.Length != (int)League.Max)
                throw new ArgumentException();

            for (int i = list.Length-1; i >= 0; i--)
            {
                //Console.WriteLine(list[i]);
                if (list[i] > 0.5)
                    return (League)(i+1);
            }
            return League.Max;
        }
    }

    /// <summary>
    /// 43.76% => 3-21-0.006-1000 ==> Error=0.668 /\ Success=43.73%
    /// 43.67% => 3-19-0.005-1000 ==> Error=?
    /// </summary>
    public class OptimizationTrier
    {
        private const int MaxThreads = 8;
        private readonly Tuple<IEnumerable<double[]>, IEnumerable<double[]>> TrainingSet;
        private readonly Task[] TrainingTasks;
        private readonly IEnumerator<NeuralNetwork.NetworkParameters> ParametersEnumerator;
        private readonly List<Tuple<NeuralNetwork, NeuralNetwork.TrainingResults, NeuralNetwork.NetworkParameters>> bestRuns;

        public OptimizationTrier(Tuple<IEnumerable<double[]>, IEnumerable<double[]>> ts)
        {
            TrainingSet = ts;
            TrainingTasks = new Task[MaxThreads];
            ParametersEnumerator = parametersGenerator().Shuffle(new Random()).Take(MaxThreads).GetEnumerator();
            bestRuns = new List<Tuple<NeuralNetwork, NeuralNetwork.TrainingResults, NeuralNetwork.NetworkParameters>>(3);

            for (int i = 0; i < MaxThreads; i++)
            {
                TrainingTasks[i] = new Task(runner, i, TaskCreationOptions.LongRunning);
            }
        }

        private IEnumerable<NeuralNetwork.NetworkParameters> parametersGenerator()
        {
            const double minTrainRate = 0.0005;
            const double maxTrainRate = 0.005;
            const double trainRateStep = 0.0005;
            const int minLayers = 3;
            const int maxLayers = 4;
            const int minNeurons = 18;
            const int maxNeurons = 22;
            const int minDepth = 1000;
            const int maxDepth = 1000;
            const int inputs = 18;
            const int outputs = 8;
            Random rand = new Random();

            for (double rate = minTrainRate; rate <= maxTrainRate; rate += trainRateStep)
            {
                for (int layers = minLayers; layers <= maxLayers; layers++)
                {
                    for (int neurons = minNeurons; neurons <= maxNeurons; neurons += 1)
                    {
                        int depth = rand.Next(minDepth, maxDepth);
                        yield return new NeuralNetwork.NetworkParameters
                        {
                            Inputs = inputs,
                            Outputs = outputs,
                            Checker = Program.checker,
                            Layers = layers,
                            MaxDepth = depth,
                            NeuronsPerLayers = neurons,
                            TrainingRate = rate
                        };
                    }
                }
            }
        }

        public NeuralNetwork Run()
        {
            foreach (var t in TrainingTasks)
                t.Start();

            Task.WaitAll(TrainingTasks);

            // Save the three bests
            Console.WriteLine("Optimization is over.");
            var ser = new XmlSerializer(typeof(NeuralNetwork));
            var logs = File.Create("bestRuns-logs.txt");
            for (int i = 0; i < bestRuns.Count; i++)
            {
                var run = bestRuns[i];
                var file = File.Create("bestRuns-network-" + i + ".xml");
                ser.Serialize(file, run.Item1);
                file.Close();
                var log = Encoding.UTF8.GetBytes(string.Format("Run {4} had params Count={0} Success={1} MSE={2:.000000} Time={3} %={5:.00}\n", 
                    run.Item2.count, run.Item2.success, run.Item2.mse*100, run.Item2.duration, i, run.Item2.success/run.Item2.count*100));
                logs.Write(log, 0, log.Length);
            }

            var best = bestRuns.OrderBy(r => -r.Item2.success).First();
            Console.WriteLine("Best network has => MSE={0:.00} /\\ Success={1:.00}", best.Item2.mse*100, ((double)best.Item2.success/best.Item2.count)*100);
            Console.WriteLine(" Parameters were => Layers={0} /\\ Neurons={1} /\\ Rate={2} /\\ Depth={3}", best.Item3.Layers, best.Item3.NeuronsPerLayers, best.Item3.TrainingRate, best.Item3.MaxDepth);
            return best.Item1;
        }

        private void runner(object state)
        {
            int id = (int)state;
            do
            {
                NeuralNetwork.NetworkParameters param;
                lock (ParametersEnumerator)
                {
                    if (!ParametersEnumerator.MoveNext())
                    {
                        // Done
                        return;
                    }
                    param = ParametersEnumerator.Current;
                    Console.WriteLine(string.Format("Id {4} Starting an optimization run : L={0} N={1} D={2} R={3}", param.Layers, param.NeuronsPerLayers, param.MaxDepth, param.TrainingRate, id));
                }

                NeuralNetwork network = new NeuralNetwork(param);
                var results = network.TrainBatch(TrainingSet.Item1.ToList(), TrainingSet.Item2.ToList(), s => Console.WriteLine("Id "+ id + " " + s));

                // Check against bests
                lock (bestRuns)
                {
                    Console.WriteLine(string.Format("Optimisation for {0} is done in {1}s. Results => MSE={2:.00}% // Success={5:.00} ({3}/{4})", id, results.duration.TotalSeconds, results.mse*100, results.success, results.count, ((double)results.success/results.count)*100));
                    if (bestRuns.Count < bestRuns.Capacity)
                    {
                        bestRuns.Add(Tuple.Create(network, results, param));

                    }
                    else
                    {
                        for(int i = 0; i < bestRuns.Count; i++)
                        {
                            if (bestRuns[i].Item2.success < results.success)
                            {
                                bestRuns[i] = Tuple.Create(network, results, param);
                            }
                        }
                    }
                    break;
                }
            } while (true);
        }
    }
}

