using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
namespace task
{
    public class DuplicatedData
    {
        private readonly MLContext mlContext;
        private readonly IDataView dataView;
        private readonly ITransformer model;

        public DuplicatedData(string dataPath)
        {
            mlContext = new MLContext();

            // Load the dataset
            dataView = mlContext.Data.LoadFromTextFile<Data>(dataPath, separatorChar: ',');

            // Define the pipeline
            var pipeline = mlContext.Transforms.Concatenate("Features", nameof(Data.Column1), nameof(Data.Column2), nameof(Data.Column3))
                .Append(mlContext.Transforms.DropColumns(nameof(Data.Label)))
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(mlContext.Transforms.NormalizeMeanVariance("Features"))
                .Append(mlContext.Transforms.ProjectToPrincipalComponents("Features", rank: 1))
                .Append(mlContext.Transforms.NormalizeMinMax("PC1"));

            // Train the model
            model = pipeline.Fit(dataView);
        }

        public void PrintDuplicatedData()
        {
            // Use the model to make predictions on the dataset
            var predictions = model.Transform(dataView);

            // Identify the most duplicated data points
            var scores = mlContext.Transforms.NormalizeMinMax("PC1").Fit(predictions).Transform(predictions);
            var indices = Enumerable.Range(0, predictions.Schema.Count)
                .Where(i => predictions.Schema[i].Name != nameof(Data.Label))
                .OrderByDescending(i => scores.GetColumn<float[]>(scores.Schema[i].Name).Average())
                .Take(5);

            Console.WriteLine("Duplicated data points:");
            foreach (var index in indices)
            {
                Console.WriteLine($"{predictions.Schema[index].Name}: {scores.GetColumn<float[]>(scores.Schema[index].Name).Average()}");
            }
        }

        private class Data
        {
            [LoadColumn(0)]
            public float Column1 { get; set; }

            [LoadColumn(1)]
            public float Column2 { get; set; }

            [LoadColumn(2)]
            public float Column3 { get; set; }

            [LoadColumn(3)]
            public float Label { get; set; }
        }
    }
    // Read Me //
    /*
     * This class uses ML.NET to find duplicate data points in a CSV file and accepts a path to the CSV file as input.
     * The top 5 duplicated data points in the file, together with their average scores, are printed by the
     * PrintDuplicatedData function. Other techniques for locating duplication could also be employed, but in this
     * pipeline all features are concatenated before PCA. This method may not work for all forms of data because it
     * assumes that all the features are continuous variables.
     */
    /*
     * To Run this code u need to use that temp in the main method in Program.cs
     * Though Replace "path/to/data.csv" with the actual path to your CSV file, and then run the Main method.
     * The PrintDuplicatedData method will print the top 5 duplicated data points in the file along with their average scores.
     */
    /*
     * // Replace "path/to/data.csv" with the actual path to your CSV file
               var duplicatedData = new DuplicatedData("path/to/data.csv");
               duplicatedData.PrintDuplicatedData();
     */
}