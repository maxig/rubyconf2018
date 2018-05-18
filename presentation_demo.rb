# For just one classifier
require 'opencv'

def classify(filename = 'conf2.jpg', scale_factor = 1.1, min_neighbors = 3)
  folder = '/Users/max/work/sumatosoft/libs/opencv/data'
  image = Cv::imread(filename, Cv::COLOR_BGR2GRAY)

  color = Cv::Scalar.new(0, 255, 255)
  classifier = Cv::CascadeClassifier.new(File.join(folder, 'haarcascades/haarcascade_frontalface_alt.xml'))
  classifier.detect_multi_scale(image, scale_factor: scale_factor, min_neighbors: min_neighbors).each do |r|
    pt1 = Cv::Point.new(r.x, r.y)
    pt2 = Cv::Point.new(r.x + r.width, r.y + r.height)
    image.rectangle!(pt1, pt2, color, thickness: 2, line_type: Cv::CV_AA)
  end

  output_filename = filename.split('.')[0...-1].join('_') + '_result.' + filename.split('.').last
  Cv::imwrite(output_filename, image)

  output_filename
end

classify('conf3.jpg')
classify('conf3.jpg', 1.01, 1)
classify('conf4.jpg', 1.01, 50)
classify('conf5.jpg', 1.5)
classify('conf6.jpg', 1.5)
classify('conf7.jpg', 1.5)
classify('conf8.jpg', 1.5)
