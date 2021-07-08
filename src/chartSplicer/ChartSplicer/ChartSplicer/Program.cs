using System;
using System.IO;

namespace ChartSplicer
{
    class Program
    {
        static void Main(string[] args)
        {

            Spliter spliter = new Spliter();
            spliter.SplitFile(".krakenUSD.csv",
                10,
                6,
                60 * 24 * 14,
                24 * 60 * 3);
        }
    }
}
