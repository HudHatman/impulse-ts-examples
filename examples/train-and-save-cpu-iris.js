const {
  NetworkBuilder: { NetworkBuilder1D },
  Layer: { LogisticLayer, ReluLayer, TanhLayer, SoftmaxLayer },
  Optimizer: { OptimizerGradientDescent, OptimizerMomentum, OptimizerAdagrad },
  Trainer: { Trainer },
} = require("impulse-ts");
const {
  DatasetBuilder: { DatasetBuilder },
  DatasetBuilderSource: { DatasetBuilderSourceCSV },
  DatasetModifier: { MinMaxScalingDatasetModifier, MissingDataScalingDatasetModifier, ShuffleDatasetModifier },
} = require("impulse-dataset-ts");
const { CalcMatrix2D } = require("impulse-math-device-ts");
const path = require("path");

const builder = new NetworkBuilder1D([4]);
builder
  .createLayer(ReluLayer, (layer) => {
    layer.setSize(10);
  })
  .createLayer(ReluLayer, (layer) => {
    layer.setSize(10);
  })
  .createLayer(LogisticLayer, (layer) => {
    layer.setSize(3);
  });

const network = builder.getNetwork();

DatasetBuilder.fromSource(DatasetBuilderSourceCSV.fromLocalFile(path.resolve(__dirname, "./data/iris_x.csv"))).then(
  async (inputDataset) => {
    console.log("Loaded iris_x.csv");
    DatasetBuilder.fromSource(DatasetBuilderSourceCSV.fromLocalFile(path.resolve(__dirname, "./data/iris_y.csv"))).then(
      async (outputDataset) => {
        const x = inputDataset.exampleAt(0);
        console.log("forward", x, x.get());
        console.log("forward", network.forward(inputDataset.data.transpose()).dims())
        //inputDataset = new MinMaxScalingDatasetModifier().apply(inputDataset);
        //console.log(inputDataset.exampleAt(0).get())
        const trainer = new Trainer(network, new OptimizerGradientDescent());
        trainer.setIterations(100);
        trainer.setLearningRate(0.01);
        trainer.setRegularization(0.1);
        console.log("cost", trainer.cost(inputDataset, outputDataset));
        console.log(inputDataset.getNumberOfExamples(), inputDataset.getExampleSize())
        trainer.setStepCallback(() => {
          //console.log(process.memoryUsage())
        });
        trainer.train(inputDataset, outputDataset);
        //console.log(inputDataset.data.calcSync((calc) => {
        //  calc.pow(2);
        //  console.log(calc.sum());
        //}).get())
        ///console.log(inputDataset.exampleAt(0).get(), inputDataset.exampleAt(0).transpose().get());
        /*console.log("forward", network.forward(inputDataset.exampleAt(0)));
        inputDataset = new MinMaxScalingDatasetModifier().apply(inputDataset);
        const trainer = new Trainer(network, new OptimizerGradientDescent());
        trainer.setIterations(1000);
        trainer.setLearningRate(0.15);
        trainer.setRegularization(0.1);
        console.log("cost", trainer.cost(inputDataset, outputDataset));
        trainer.train(inputDataset, outputDataset);
        await network.save(path.resolve(__dirname, "./data/iris.json"));
        console.log("forward", network.forward(inputDataset.exampleAt(0)), outputDataset.exampleAt(0));*/
      }
    );
  }
);
