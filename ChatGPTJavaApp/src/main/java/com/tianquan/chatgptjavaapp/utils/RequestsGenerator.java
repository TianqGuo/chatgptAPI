package com.tianquan.chatgptjavaapp.utils;
//import okhttp3.;

import okhttp3.OkHttpClient;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.client.RestTemplate;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

public class RequestsGenerator {
    private static final String apiUrl = "https://api.openai.com/v1/chat/completions";
    @Value("${apikey}")
    private String apiKey;
    private final RestTemplate restTemplate = new RestTemplate();

    public String generateChat(String prompt) throws IOException {
//        HttpHeaders headers = new HttpHeaders();
//        headers.setContentType(MediaType.APPLICATION_JSON);
//        headers.set("Authorization", "Bearer " + apiKey);
//        System.out.println("apikey is: " + apiKey);
        OkHttpClient client = new OkHttpClient();
        String modelId = "gpt-3.5-turbo";

//        String requestJson = "{\n" +
//                "  \"model\": \"gpt-3.5-turbo\",\n" +
//                "  \"input\": \"" + prompt + "\",\n" +
//                "}";
        String requestJson = "{\"model\": \"" + modelId + "\", \"messages\": [{\"role\": \"system\", \"content\": \"You are a helpful assistant.\"}, {\"role\": \"user\", \"content\": \"" + prompt + "\"}]}";

        HttpURLConnection connection = (HttpURLConnection) new URL(apiUrl).openConnection();

        connection.setRequestMethod("POST");
        connection.setRequestProperty("Content-Type", "application/json");
        connection.setRequestProperty("Authorization", "Bearer " + apiKey);

        connection.setDoOutput(true);
        connection.getOutputStream().write(requestJson.getBytes());

        BufferedReader reader = new BufferedReader(new InputStreamReader(connection.getInputStream()));
        String response = reader.lines()
                .reduce((a, b) -> a + b)
                .get();

        System.out.println(response);

        return response;

    }
}
