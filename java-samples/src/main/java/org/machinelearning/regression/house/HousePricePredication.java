package org.machinelearning.regression.house;

import smile.data.Tuple;
import smile.data.type.DataType;
import smile.data.type.StructField;
import smile.data.type.StructType;
import smile.regression.LinearModel;

import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.stream.Collectors;

public class HousePricePredication {

    public static void main(String[] args) throws Exception {
        Path path = Paths.get(args[0]);

        LinearModel housePredication = loadModel(path);
        System.out.println(housePredication);

        String example = "335000.0,2.0,2.0,1350,2560,1.0,0,0,3,1350,0,1976,0";
        StructType scoreSchema = createScoreSchema();
        System.out.println("input features " + Arrays.asList(scoreSchema.fields()).stream().skip(1).collect(Collectors.toList()));
        System.out.println("Eg " + example);


        new BufferedReader(new InputStreamReader(System.in))
                .lines()
                .filter(line -> !line.trim().isEmpty())
                .map(line -> line.split(","))
                .map(row -> toDouble(row))
                .map(vector -> Tuple.of(vector, createScoreSchema()))
                .forEach(features -> predict(housePredication, features));


    }

    private static double[] toDouble(String[] row) {
        return Arrays.asList(row).stream().mapToDouble(Double::parseDouble).toArray();
    }

    private static void predict(LinearModel housePredication, Tuple features) {
        double housePrice = housePredication.predict(features);
        System.out.println(String.format("Expected price of house is %s", housePrice));
    }

    private static LinearModel loadModel(Path path) throws IOException, ClassNotFoundException {
        System.out.println("Loading model from " + path);
        FileInputStream fin = new FileInputStream(path.toFile());
        ObjectInputStream ins = new ObjectInputStream(fin);
        LinearModel housePredication = (LinearModel) ins.readObject();
        System.out.println("Model file loaded ");
        return housePredication;
    }

    private static StructType createScoreSchema() {
        return new StructType(
                field("price", DataType.of(double.class)),
                field("bedrooms", DataType.of(double.class)),
                field("bathrooms", DataType.of(double.class)),
                field("sqft_living", DataType.of(double.class)),
                field("sqft_lot", DataType.of(double.class)),
                field("floors", DataType.of(double.class)),
                field("waterfront", DataType.of(double.class)),
                field("view", DataType.of(double.class)),
                field("condition", DataType.of(double.class)),
                field("sqft_above", DataType.of(double.class)),
                field("sqft_basement", DataType.of(double.class)),
                field("yr_built", DataType.of(double.class)),
                field("yr_renovated", DataType.of(double.class))
        );

    }

    private static StructField field(String name, DataType colType) {
        return new StructField(name, colType);
    }
}
