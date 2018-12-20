import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# mnist data setini import ettik. Nokta operatoru ile tensorflow kutuphanesi içindeki modullere erisiyoruz.

mnist = input_data.read_data_sets("data/MNIST/",one_hot = True)
# bu fonksiyon ile mnist data setini alabiliyoruz. mnist data setini 2 şeyle alıyoruz resimler ve değerleri. Değerlerini de one_hot yöntemi ile alıyoruz.

x = tf.placeholder(tf.float32, [None, 784])
#Gelen resim verilerini atayacağımız place holder, bunlar yer tutuculardır. 28*28 boyutlarında mnist dataseti bu 784 piksel olduğu için son parametreye yazdık.

y_true = tf.placeholder(tf.float32, [None, 10])
#Doğru değerler için bunu oluşturduk. Mnist data setinde 10 sınıf olduğu için son parametreye yazdık bunu.

w = tf.Variable(tf.zeros([784,10]))
#Eğitilecek parametreleri tf.Variable yaparak tanımlıyoruz. Bu fonksiyon tüm elemanları sıfır olan, 784 genişliğinde ve 10 uzunluğunda bir matris oluşturuyor.

b= tf.Variable(tf.zeros([10]))
#bias da weight gibi eğitim esnasında optimize edilecek.


logits = tf.matmul(x, w) + b
#matrislerde çarpma işlemi yapmak için tf.matmul yapıyoruz.

y = tf.nn.softmax(logits)
#gelen logits değerlerini softmax aktivasyon fonksiyonundan geçirerek 0-1 arasında sıkıştırıyoruz.En büyük değer hangisiyse ona göre tahmin yapacak.

xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
loss = tf.reduce_mean(xent)
#loss fonksiyonu ile modelin ne kadar doğru ne kadar yanlış olduğunu hesaplayalım. loss fonksiyonu olarak cross entropy kullanıcaz. Ve bu değerin ortalamasını alıyoruz.

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_true,1))
#tf.equal yaptığımız tahmin doğru mu yanlış mı ona göre boolean bir değer döndürücek.Tahmini almak için tf.argmax kullanıyoruz.
# argmax fonksiyonu vektör içindeki en yüksek değerin kaçıncı sırada olduğunu veriyor.

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#ortalama alma işlemi

optimize = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
#optimizasyon işlemi yapılıyor, parametre olarak learning rate alıyor.

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 128
#mnist data setinde 70.000 resim var, bunu parça parça alıp eğiteceğiz.

def training_step(iterations):
    for i in range(iterations):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        feed_dict_train = {x: x_batch, y_true: y_batch }
        sess.run(optimize, feed_dict=feed_dict_train)


def test_accuracy():
    feed_dict_test = {x: mnist.test.images, y_true: mnist.test.labels}
    acc = sess.run(accuracy, feed_dict=feed_dict_test)
    print("Testing accuracy:", acc)


training_step(2000)
test_accuracy()

