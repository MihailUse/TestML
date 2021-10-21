using System;
using static Tensorflow.Binding;

namespace TestML
{
    class Program
    {
        static void Main(string[] args)
        {
            var hello = tf.constant("Hello, TensorFlow!");
			Console.WriteLine("\n\n");

            var cnn = new ClassificationCNN();
            cnn.Run();
        }
    }
}
