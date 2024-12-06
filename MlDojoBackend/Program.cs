using Microsoft.ML;
using MlDojoBackend;

var ctx = new MLContext();

//change the path if on windows. needs refactoring.
var trainingData = ctx.Data
    .LoadFromTextFile<ModelInput>(path: "../../../trainingData/sentiment_data.csv", 
        hasHeader: true, 
        separatorChar: ',');

var pipeline = ctx.Transforms.Text
    .FeaturizeText("Features", nameof(ModelInput.Text))
    .Append(ctx.BinaryClassification.Trainers
        .LbfgsLogisticRegression(nameof(ModelInput.Label), "Features"));

ITransformer trainedModel = pipeline.Fit(trainingData);

var predictionEngine = ctx.Model
    .CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);

var sampleStatement = new ModelInput() { Text = "The service was outstanding" };
var prediction = predictionEngine.Predict(sampleStatement);

Console.WriteLine($"Prediction: {prediction.IsPositive}, Probability: {prediction.Probability}");
