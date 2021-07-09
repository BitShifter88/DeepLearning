using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace ChartSplicer
{
    class Spliter
    {
        public void SplitFile(string file, int resolutionMin, int spliceStrade, int intervalMin, int predictionMin)
        {
            string trainingFolder = "data/training";
            string testFolder = "data/test";
            string validationFolder = "data/validation";

            Directory.Delete(trainingFolder, true);
            Directory.Delete(testFolder, true);
            Directory.Delete(validationFolder, true);

            Directory.CreateDirectory(trainingFolder);
            Directory.CreateDirectory(testFolder);
            Directory.CreateDirectory(validationFolder);


            var trades = ParseTrades(file, resolutionMin);

            int tradesCount = (int)((double)trades.Count * 0.8);

            int testStart = (int)((double)trades.Count * 0.8);
            int testCount = (int)((double)trades.Count * 0.1);

            int validationStart = (int)((double)trades.Count * 0.9);
            int validationCount = (int)((double)trades.Count * 0.1);

            List<Trade> training = trades.GetRange(0, tradesCount);
            List<Trade> test = trades.GetRange(testStart, testCount);
            List<Trade> validation = trades.GetRange(validationStart, validationCount);

            Console.WriteLine("Splitting training data");
            Split(resolutionMin, spliceStrade, intervalMin, predictionMin, training, trainingFolder);
            Console.WriteLine("Splitting test data");
            Split(resolutionMin, spliceStrade, intervalMin, predictionMin, test, testFolder);
            Console.WriteLine("Splitting validation data");
            Split(resolutionMin, spliceStrade, intervalMin, predictionMin, validation, validationFolder);

            Console.WriteLine(TradeSplit._down);
            Console.WriteLine(TradeSplit._up);
            Console.WriteLine(TradeSplit._middle);
        }

        private static void Split(int resolutionMin, int stride, int intervalMin, int predictionMin, List<Trade> data, string folder)
        {
            List<TradeSplit> splits = new List<TradeSplit>();

            for (int i = 0; i < data.Count; i += stride)
            {
                List<double> prices = new List<double>();
                int endIndex = i + intervalMin / resolutionMin;
                int predictionIndex = endIndex + predictionMin / resolutionMin;
                if (data.Count <= predictionIndex)
                    break;

                for (int j = i; j < endIndex; j++)
                {
                    Trade trade = data[j];
                    prices.Add(trade.Price);
                }

                double prediction = data[predictionIndex].Price;

                TradeSplit split = new TradeSplit() { Prices = prices, Prediction = prediction };
                
                split.Normalize();
                splits.Add(split);
            }

            List<TradeSplit> buffer = new List<TradeSplit>();
            int counter = 0;
            foreach (TradeSplit s in splits)
            {
                buffer.Add(s);

                if (buffer.Count == 1000)
                {
                    WriteBuffer(folder, buffer, counter);
                    buffer.Clear();
                }
                counter++;
            }
            WriteBuffer(folder, buffer, counter);

        }

        private static void WriteBuffer(string folder, List<TradeSplit> buffer, int counter)
        {
            string filePath = Path.Combine(folder, counter.ToString() + ".json");
            string splitJson = JsonConvert.SerializeObject(buffer);
            File.WriteAllText(filePath, splitJson);
            Console.WriteLine(buffer[0].Prices.Count);


            // using (BinaryWriter bw = new BinaryWriter(new FileStream(filePath, FileMode.Create)))
            // {
            //     bw.Write(buffer.Count);

            //     foreach (var tradeSplit in buffer)
            //     {
            //         bw.Write(tradeSplit.Prices.Count);
            //         bw.Write(tradeSplit.Prediction);

            //         foreach (var trade in tradeSplit.Prices)
            //         {
            //             bw.Write(trade);
            //         }
            //     }
            // }
        }

        private List<Trade> ParseTrades(string file, int resolutionMin)
        {
            List<Trade> trades = new List<Trade>();

            double volumeCounter = 0;
            DateTime lastTrade = DateTime.MinValue;

            Console.WriteLine("Loading trades...");
            using (StreamReader sr = new StreamReader(new FileStream(file, FileMode.Open)))
            {
                int lineCount = 0;
                while (!sr.EndOfStream)
                {
                    string line = sr.ReadLine();
                    string[] split = line.Split(',');

                    DateTime time = UnixTimeStampToDateTime(int.Parse(split[0]));
                    double price = double.Parse(split[1], CultureInfo.InvariantCulture);
                    double volume = double.Parse(split[2], CultureInfo.InvariantCulture);

                    volumeCounter += volume;

                    TimeSpan timeDelta = time.Subtract(lastTrade);
                    //if (timeDelta.Minutes > resolutionMin)
                    //{
                    //    Console.WriteLine(lineCount.ToString() + " - " + timeDelta.Minutes + " - " + time.Year);
                    //}

                    if (timeDelta.Minutes >= resolutionMin)
                    {
                        trades.Add(new Trade(time, price, volumeCounter));
                        volumeCounter = 0;
                        lastTrade = time;
                    }

                    lineCount++;
                }
            }
            Console.WriteLine("Loaded!");
            return trades;
        }

        public DateTime UnixTimeStampToDateTime(double unixTimeStamp)
        {
            // Unix timestamp is seconds past epoch
            System.DateTime dtDateTime = new DateTime(1970, 1, 1, 0, 0, 0, 0, System.DateTimeKind.Utc);
            dtDateTime = dtDateTime.AddSeconds(unixTimeStamp).ToLocalTime();
            return dtDateTime;
        }
    }

    class TradeSplit
    {
        public List<double> Prices { get; set; } = new List<double>();
        public double Prediction { get; set; }
        public int Up { get; set; }
        public int Down { get; set; }
        public double Min { get; set; }
        public double Max { get; set; }

        public void Normalize()
        {
            List<double> data = Prices.ToList();
            //data.Add(Prediction);

            var normalized = NormalizeData(data, 0.0, 1.0, Prediction).ToList();
            //Prediction = normalized.Last();
            //normalized.Remove(normalized.Last());
            Prices = normalized;

            if (Prediction > 2)
            {

            }
            if (Prediction < 0)
            {

            }
        }

        public static int _up = 0;
        public static int _down = 0;
        public static int _middle = 0;
        private double[] NormalizeData(List<double> data, double min, double max, double prediction)
        {
            List<double> diffs = new List<double>();

            for (int i = 0; i < data.Count-1; i++)
            {
                double value = data[i];
                double nextValue = data[i+1];
                double diff = value - nextValue;
                diffs.Add(diff);
            }

            double dataMax = diffs.Max();
            double dataMin = diffs.Min();
            //data.Add(prediction);
            double range = dataMax - dataMin;

            double lastPrice = data.Last();

            double percent = (prediction - lastPrice) / lastPrice;
            //percent *= 100;
            Prediction = percent;

            double cutOff = 0.01;
            if (percent >= 0)
                {
                    Up = 1;
                    _up++;
                }
                else if (percent <= 0)
                _down++;
            // if (percent >= cutOff)
            // {
            //     Up = 1;
            //     _up++;
            // }
            // else if (percent <= -cutOff)
            // {
            //     Up = 0;
            //     _down++;
            // }
            // else
            // {
            //     Up = 2;
            //     _middle++;
            // }

            Min = dataMin;
            Max = dataMax;

            var normalized = diffs
                .Select(d => (d - dataMin) / range)
                .Select(n => (double)((1 - n) * min + n * max))
                .ToArray();
            
            return normalized;
        }
    }

    class Trade
    {
        public DateTime Time { get; set; }
        public double Price { get; set; }
        public double Volume { get; set; }

        public Trade(DateTime time, double price, double volume)
        {
            Time = time;
            Price = price;
            Volume = volume;
        }
    }
}
