using System;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.Utils;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using Tensorflow.NumPy;
using System.IO;

namespace TestML
{
	class ClassificationCNN
	{
		const string dataDir = ".\\dataset";

		const int imgHeight = 150;
		const int imgWidth = 150;

		int nClasses = 2;
		int nChannels = 3;

		int epochs = 10;
		int batchSize = 64;
		float learningRate = 0.0002f;


		public void PrepareData()
		{
			LoadDataset();

			var datasetDir = Path.Join(dataDir, "cats_and_dogs_filtered");
			var trainDir = Path.Join(datasetDir, "train");
			var validationDir = Path.Join(datasetDir, "validation");

		}

		private void LoadDataset()
		{
			Directory.CreateDirectory(dataDir);

			Web.Download("https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip", dataDir, "cats_and_dogs.zip");
			Compress.UnZip(Path.Join(dataDir, "cats_and_dogs.zip"), dataDir);
		}
	}
}
