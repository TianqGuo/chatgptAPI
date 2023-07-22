import React, {Component} from 'react';
import { Button, Input, Card, Form, InputNumber, Tabs} from 'antd';
import axios from 'axios';

const ASK_QUETIONS_URL = 'http://127.0.0.1:5000/api/v1/chat/'
const QUOTE_MODEL_URL = 'http://127.0.0.1:5000/api/v1/model/prediction'
const { TextArea } = Input;

class Physician extends Component {
    constructor(props) {
        super(props);
        // define state for the components states
        this.state = {
            currentQuestion: null,
            currentQuote: null,
            currentAnswer: "Your answer will be displayed here",
            gender: 0,
            age: 80,
            hypertension: 0,
            heartdisease: 1,
            smokingHistory: 4,
            bmi: 25.19,
            hbA1cLevel: 6.6,
            bloodGlucoseLevel: 140,
            currentPrediction: "Your prediction will be displayed here",
        }
        this.formRef = React.createRef();
    }

    onClickExample1 = async () => {
        console.log("clicked example 1");
        await this.setState({ gender: 0 });
        this.formRef.current.setFieldsValue({
            gender: 0,
        });
        await this.setState({ age: 80 });
        this.formRef.current.setFieldsValue({
            age: 80,
        });
        await this.setState({ hypertension: 0 });
        this.formRef.current.setFieldsValue({
            hypertension: 0,
        });
        await this.setState({ heartdisease: 1 });
        this.formRef.current.setFieldsValue({
            heartdisease: 1,
        });
        await this.setState({ smokingHistory: 4 });
        this.formRef.current.setFieldsValue({
            smokingHistory: 4,
        });
        await this.setState({ bmi: 25.19 });
        this.formRef.current.setFieldsValue({
            bmi: 25.19,
        });
        await this.setState({ hbA1cLevel: 6.6 });
        this.formRef.current.setFieldsValue({
            hbA1cLevel: 6.6,
        });
        await this.setState({ bloodGlucoseLevel: 140 });
        this.formRef.current.setFieldsValue({
            bloodGlucoseLevel: 140,
        });
        console.log("Data set to example 1");
        // console.log(this.state.gender);
    }

    onClickExample2 = async () => {
        console.log("clicked example 2");
        await this.setState({ gender: 1 });
        this.formRef.current.setFieldsValue({
            gender: 1,
        });
        await this.setState({ age: 90 });
        this.formRef.current.setFieldsValue({
            age: 90,
        });
        await this.setState({ hypertension: 1 });
        this.formRef.current.setFieldsValue({
            hypertension: 1,
        });
        await this.setState({ heartdisease: 1 });
        this.formRef.current.setFieldsValue({
            heartdisease: 1,
        });
        await this.setState({ smokingHistory: 12 });
        this.formRef.current.setFieldsValue({
            smokingHistory: 12,
        });
        await this.setState({ bmi: 40 });
        this.formRef.current.setFieldsValue({
            bmi: 40,
        });
        await this.setState({ hbA1cLevel: 10 });
        this.formRef.current.setFieldsValue({
            hbA1cLevel: 10,
        });
        await this.setState({ bloodGlucoseLevel: 260 });
        this.formRef.current.setFieldsValue({
            bloodGlucoseLevel: 260,
        });
        console.log("Data set to example 2");
        // console.log(this.state.gender);
    }

    onClickQuetions = async () => {
        console.log("clicked ask quetions");

        // const response = await this.onAskQuestions();
        // // console.log(response);
        // this.setState({ currentAnswer: response });
        // console.log(this.state.currentAnswer);
        // // console.log(this.state.currentAnswer.created)

        await this.onAskQuestions();
        console.log(this.state.currentAnswer);
    }

    onSetAnswers = async (response) => {
        await this.setState({ currentAnswer : response });
        console.log(this.state.currentAnswer);
    }

    onClickModels = async () => {
        console.log("clicked quote models");

        const response = await this.onQuoteModels();
        console.log(response);
    }

    onChangeQuestions = async (e) => {
        await this.setState({ currentQuestion : e.target.value });
        console.log(this.state.currentQuestion);
    }

    onChangeModels = async e => {
        await this.setState({ currentQuote : e.target.value });
        console.log(this.state.currentQuote);
    }

    onChangeGender = async (e) => {
        await this.setState({ gender : e });
        console.log(this.state.gender);
    }

    onChangeAge = async e => {
        await this.setState({ age : e});
        console.log(this.state.age);
    }

    onChangeHypertension = async e => {
        await this.setState({ hypertension : e });
        console.log(this.state.hypertension);
    }

    onChangeHeartdisease = async e => {
        await this.setState({ heartdisease : e });
        console.log(this.state.heartdisease);
    }

    onChangeSmokingHistory = async e => {
        await this.setState({ smokingHistory : e });
        console.log(this.state.smokingHistory);
    }

    onChangeBmi = async e => {
        await this.setState({ bmi : e });
        console.log(this.state.bmi);
    }

    onChangeHbA1cLevel = async e => {
        await this.setState({ hbA1cLevel : e });
        console.log(this.state.hbA1cLevel);
    }

    onChangeBloodGlucoseLevel = async e => {
        await this.setState({ BloodGlucoseLevel : e });
        console.log(this.state.BloodGlucoseLevel);
    }

    // onAskQuestions = async () => {
    //     return fetch(`${ASK_QUETIONS_URL}${this.state.currentQuestion}`, {
    //         method: 'GET',
    //         headers: {
    //         'Content-Type': 'application/json',
    //         },
    //     }).then((response) => {
    //         if (response.status !== 200) {
    //             throw Error('Fail to request the answer from the server');
    //         }

    //         return response.json(); // Return the promise directly
    //     }).then((result) => {
    //         console.log(result);
    //         this.setState({ currentAnswer: result }); // Set the result when the promise resolves
    //         // console.log(this.state.currentAnswer);
    //         return result; // Return the result value
    //     });
    // }

    onAskQuestions = () => {
        axios.get(`${ASK_QUETIONS_URL}${this.state.currentQuestion}`, {
            headers: {
            'Content-Type': 'application/json',
            },
        }).then((response) => {
            if (response.status !== 200) {
                throw Error('Fail to request the answer from the server');
            }

            
            console.log(response);
            this.setState({ currentAnswer: response.data.text}); // Set the result when the promise resolves
            // console.log(this.state.currentAnswer);
            return response; // Return the result value
        });
    }

    onClickPrediction = () => {
        var data = [[this.state.gender, this.state.age, this.state.hypertension, this.state.heartdisease, this.state.smokingHistory, this.state.bmi, this.state.hbA1cLevel, this.state.bloodGlucoseLevel]]
        var matrix = { 'data': data }
        console.log(matrix);
        console.log(JSON.stringify(matrix));
        axios.post(`${QUOTE_MODEL_URL}`, {
            headers: {
            'Content-Type': 'application/json',
            },
            body: JSON.stringify(matrix)
        }).then((response) => {
            if (response.status !== 200) {
                throw Error('Fail to request the answer from the server');
            }

            
            // console.log(response);
            this.setState({ currentPrediction: response.data.prediction}); // Set the result when the promise resolves
            console.log(this.state.currentPrediction);
            return response; // Return the result value
        });
    }

    onQuoteModels = async (data) => {
        return fetch(`${QUOTE_MODEL_URL}`, {
            method: 'post',
            headers: {
            'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        }).then((response) => {
            if (response.status !== 200) {
                throw Error('Fail to request the prediction from the server');
            }

            return response.json();
        })
    }

    render() {
        return (
            <div className="Main" style={{fontWeight: "bold"}}>
                <Tabs defaultActiveKey = "prediction">
                    <Tabs.TabPane tab="Prediction" key="prediction">
                        IntelliHealth Prediction
                        <p style={{fontWeight: "normal", fontSize: "15px"}}>
                            Please input the following information to get a prediction of your risk of getting diabetes.
                        </p>

                        <Form ref={this.formRef}>
                            <Form.Item label="Gender" name="gender" initialValue={0} rules={[{ required: true, message: 'Please input valid numbers'}]}>
                                <InputNumber min={0} max={1} onChange={this.onChangeGender} precision={0}/>
                            </Form.Item>
                            <Form.Item label="Age" name="age" initialValue={80} rules={[{ required: true, message: 'Please input valid numbers'}]}>
                                <InputNumber min={0} max={100} onChange={this.onChangeAge} precision={0}/>
                            </Form.Item>
                            <Form.Item label="Hypertension" name="hypertension" initialValue={0} rules={[{ required: true, message: 'Please input valid numbers'}]}>
                                <InputNumber min={0} max={1} onChange={this.onChangeHypertension} precision={0}/>
                            </Form.Item>
                            <Form.Item label="Heart Disease" name="heartdisease" initialValue={1} rules={[{ required: true, message: 'Please input valid numbers'}]}>
                                <InputNumber min={0} max={1} onChange={this.onChangeHeartdisease} precision={0}/>
                            </Form.Item>
                            <Form.Item label="Smoking History" name="smokingHistory" initialValue={4} rules={[{ required: true, message: 'Please input valid numbers'}]}>
                                <InputNumber min={0} max={100} onChange={this.onChangeSmokingHistory} precision={2}/>
                            </Form.Item>
                            <Form.Item label="BMI" name="bmi" initialValue={25.19} rules={[{ required: true, message: 'Please input valid numbers'}]}>
                                <InputNumber min={0} max={100} onChange={this.onChangeBmi} precision={2}/>
                            </Form.Item>
                            <Form.Item label="HbA1c Level" name="hbA1cLevel" initialValue={6.6} rules={[{ required: true, message: 'Please input valid numbers'}]}>
                                <InputNumber min={0} max={100} onChange={this.onChangeHbA1cLevel} precision={2}/>
                            </Form.Item>
                            <Form.Item label="Blood Glucose Level" name="bloodGlucoseLevel" initialValue={140} rules={[{ required: true, message: 'Please input valid numbers'}]}>
                                <InputNumber min={0} max={600} onChange={this.onChangeBloodGlucoseLevel} precision={2}/>
                            </Form.Item>

                        </Form>

                        <Button  type="primary" htmlType="submit" onClick={this.onClickExample1} style={{ margin: 20 }}> Example 1 </Button>

                        <Button  type="primary" htmlType="submit" onClick={this.onClickExample2} style={{  margin: 20 }}> Example 2 </Button>

                        <p></p>

                        <Button  type="primary" htmlType="submit" onClick={this.onClickPrediction}> Get Prediction </Button>





                        <div>
                            <Card title="Prediction: " bordered={false} style={{ width: 800, margin: 20}}>
                                <p >
                                    {this.state.currentPrediction}
                                </p>
                            </Card>

                        </div>
                    </Tabs.TabPane>

                    <Tabs.TabPane tab="Chatbot" key="chatbot">
                        IntelliHealth ChatBot
                        <p style={{fontWeight: "normal", fontSize: "15px"}}>
                            If you have any question, feel free to ask any questions about the diabetes
                        </p>


                        <TextArea showCount maxLength={100} onChange={this.onChangeQuestions} style={{ width: 800, height: 350 }} />
                        <p>

                        </p>

                        <Button type="primary" htmlType="submit" onClick={this.onClickQuetions}> Send it </Button>

                        <p>

                        </p>

                        <div>
                            <Card title="ChatBot Response: " bordered={false} style={{ width: 800, margin: 20 }}>
                                <p >
                                    {this.state.currentAnswer}
                                </p>
                            </Card>

                        </div>
                    </Tabs.TabPane>
                </Tabs>
            </div>
        );
    }
}
export default Physician;