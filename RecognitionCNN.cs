using System;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.Utils;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using System.IO;
using System.Collections.Generic;
using Tensorflow.NumPy;

namespace TestML
{
	class ClassificationCNN
	{
		//Dataset
		IDatasetV2 train_ds, val_ds;
		string data_dir = ".\\dataset";
		int n_classes = 2;

		//Image
		const int img_height = 150;
		const int img_width = 150;
		Shape img_dim = (img_height, img_width);
		
		//Model
		Model model;
		int epochs = 10;
		int batch_size = 64;
		float learning_rate = 0.0002f;

		public void Run()
		{
			tf.enable_eager_execution();

			PrepareData();
			BuildModel();
			Train();
		}

		public void BuildModel()
		{
			var layers = keras.layers;
			model = keras.Sequential(new List<ILayer>
			{
				layers.Rescaling(1.0f / 255, input_shape: (img_height, img_width, 3)),

				layers.Conv2D(16, 3, padding: "same", activation: "relu"),
				layers.MaxPooling2D(),

				layers.Conv2D(32, 3, padding: "same", activation: "relu"),
				layers.MaxPooling2D(),

				layers.Conv2D(64, 3, padding: "same", activation: "relu"),
				layers.MaxPooling2D(),

				layers.Flatten(),
				layers.Dense(128, activation: "relu"),
				layers.Dense(1)
			});

			model.compile(optimizer: keras.optimizers.Adam(learning_rate),
				loss: keras.losses.CategoricalCrossentropy(from_logits: true),
				metrics: new[] { "accuracy" });

			model.summary();
		}

		public void Train()
		{
			model.fit(train_ds, epochs: epochs);
		}

		public void PrepareData()
		{
			LoadDataset();

			var dataset_dir = Path.Join(data_dir, "cats_and_dogs_filtered");

			//convert to tensor
			train_ds = keras.preprocessing.image_dataset_from_directory(
				directory: Path.Join(dataset_dir, "train"),
				validation_split: 0.0f,
				subset: "training",
				seed: 123,
				image_size: img_dim,
				batch_size: batch_size
			);

			val_ds = keras.preprocessing.image_dataset_from_directory(
				directory: Path.Join(dataset_dir, "validation"),
				validation_split: 1.0f,
				subset: "validation",
				seed: 123,
				image_size: img_dim,
				batch_size: batch_size
			);

			train_ds = train_ds.shuffle(1000).prefetch(buffer_size: -1);
			val_ds = val_ds.prefetch(buffer_size: -1);

			//train_ds.map(x => tf.cast(x, tf.float32));
			//val_ds.map(x => tf.cast(x, tf.float32));

			foreach (var (img, label) in train_ds)
			{
				print($"images: {img.shape}");
				print($"dtype: {img.dtype}");
				print($"labels: {label.numpy()}");
			}
		}

		private void LoadDataset()
		{
			Directory.CreateDirectory(data_dir);

			Web.Download("https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip", data_dir, "cats_and_dogs.zip");
			Compress.UnZip(Path.Join(data_dir, "cats_and_dogs.zip"), data_dir);
		}
	}
}
