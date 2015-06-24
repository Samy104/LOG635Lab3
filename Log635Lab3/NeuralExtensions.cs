using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural
{
    public static class NeuralExtensions
    {
        public static double Sigmoid(this double val)
        {
            return 1 / (1 + Math.Exp(-val));
        }

        public static double InvSigmoid(this double val)
        {
            return Math.Log(-val / (val - 1));
        }

        public static List<T> EmptyList<T>(this T val)
        {
            var list = new List<T>();
            list.Add(val);
            return list;
        }

        public static T[] SimpleArray<T>(this T val)
        {
            return new T[] { val };
        }

        public static double[] Normalize(this double[] list, double[] mean, double[] stdev)
        {
            bool incompatibleData = false;
            double[] output = new double[list.Length];
            double meanNorm = 0;
            for (int i = 0; i < list.Length; i++)
            {
                if (list[i] == -1)
                {
                    incompatibleData = true;
                    output[i] = double.NaN;
                }
                output[i] = (list[i] - mean[i]) / stdev[i];
                meanNorm += output[i];
            }
            if (incompatibleData)
            {
                meanNorm /= list.Length;
                for (int i = 0; i < output.Length; i++)
                    if (double.IsNaN(output[i]))
                        output[i] = meanNorm;
            }

            return output;
        }

        public static void ForEach<T>(this IEnumerable<T> en, Action<T> ac)
        {
            foreach (var v in en)
            {
                ac(v);
            }
        }

        public static void ForEach<T>(this IEnumerable<T> en, Action<T, int> ac)
        {
            int i = 0;
            foreach (var v in en)
            {
                ac(v, i);
                i++;
            }
        }

        public static IEnumerable<Tuple<T, S>> Tuplify<T, S>(this IEnumerable<T> orig, IEnumerable<S> other)
        {
            var orEnum = orig.GetEnumerator();
            var otEnum = other.GetEnumerator();

            while (orEnum.MoveNext() && otEnum.MoveNext())
            {
                yield return Tuple.Create(orEnum.Current, otEnum.Current);
            }
        }

        public static IEnumerable<T> Shuffle<T>(this IEnumerable<T> source, Random rng)
        {
            T[] elements = source.ToArray();
            for (int i = elements.Length - 1; i >= 0; i--)
            {
                int swapIndex = rng.Next(i + 1);
                yield return elements[swapIndex];
                elements[swapIndex] = elements[i];
            }
        }
    }
}
