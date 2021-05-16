package org.machinelearning.regression.house;

import org.apache.commons.csv.CSVFormat;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.data.type.DataType;
import smile.data.type.StructField;
import smile.data.type.StructType;
import smile.regression.LinearModel;
import smile.regression.OLS;

import java.io.*;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.function.Function;

public class HousePriceModelTraining {

    public static void main(String[] args) throws Exception {
        URL u = HousePriceModelTraining.class.getResource("/houseprice/data.csv");
        Path p = Paths.get(u.toURI());

        System.out.println(p);

        Function<String, Path> pipeline = read
                .andThen(dropCols)
                .andThen(trainModel)
                .andThen(saveModel);
        Path modelLocation = pipeline.apply(p.toString());
        System.out.println(String.format("Model is saved @ %s", modelLocation));
    }

    private static StructType createInputSchema() {
        StructType schema = new StructType(
                field("date", DataType.of(String.class)),
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
                field("yr_renovated", DataType.of(double.class)),
                field("street", DataType.of(String.class)),
                field("city", DataType.of(String.class)),
                field("statezip", DataType.of(String.class)),
                field("country", DataType.of(String.class))
        );
        return schema;
    }


    private static StructField field(String name, DataType colType) {
        return new StructField(name, colType);
    }

    static Function<String, DataFrame> read = f -> {
        StructType schema = createInputSchema();
        try {
            return smile.io.Read.csv(Paths.get(f), CSVFormat.DEFAULT, schema);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    };

    static Function<DataFrame, DataFrame> dropCols = df -> df.drop("date", "street", "city", "statezip", "country");


    static Function<DataFrame, LinearModel> trainModel = df -> {
        Formula formula = Formula.lhs("price");
        return OLS.fit(formula, df);
    };


    static Function<LinearModel, Path> saveModel = model -> {
        try {
            File tempFile = File.createTempFile("house_", "_model");
            FileOutputStream fos = new FileOutputStream(tempFile);
            ObjectOutputStream os = new ObjectOutputStream(fos);
            os.writeObject(model);
            return tempFile.toPath();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    };


}
