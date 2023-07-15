import React, {Component} from 'react';
import { Button, Input, Card, Form, InputNumber} from 'antd';
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
            hypentension: 0,
            heartdisease: 1,
            smokingHistory: 4,
            bmi: 25.19,
            hbA1cLevel: 6.6,
            bloodGlucoseLevel: 140,
            currentPrediction: null,
        }
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

    onChangeHypentension = async e => {
        await this.setState({ hypentension : e });
        console.log(this.state.hypentension);
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
        var data = [[this.state.gender, this.state.age, this.state.hypentension, this.state.heartdisease, this.state.smokingHistory, this.state.bmi, this.state.hbA1cLevel, this.state.bloodGlucoseLevel]]
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
                IntelliHealthPrediction
                <p style={{fontWeight: "normal", fontSize: "15px"}}>
                    Please input the following information to get a prediction of your risk of getting diabetes.
                </p>

                <Form >
                    <Form.Item label="Gender" name="gender" rules={[{ required: true, message: 'Please input valid numbers'}]}>
                        <InputNumber min={0} onChange={this.onChangeGender} precision={2}/>
                        {/* <Input onChange={this.onChangeAreaCode}/> */}
                    </Form.Item>
                    <Form.Item label="Age" name="age" rules={[{ required: true, message: 'Please input valid numbers'}]}>
                        <InputNumber min={0} onChange={this.onChangeAge} precision={2}/>
                        {/* <Input onChange={this.onChangeSevenDigits}/> */}
                    </Form.Item>
                    <Form.Item label="Hypertension" name="hypertension" rules={[{ required: true, message: 'Please input valid numbers'}]}>
                        <InputNumber min={0} onChange={this.onChangeHypentension} precision={2}/>
                    </Form.Item>
                    <Form.Item label="Heart Disease" name="heartdisease" rules={[{ required: true, message: 'Please input valid numbers'}]}>
                        <InputNumber min={0} onChange={this.onChangeHeartdisease} precision={2}/>
                    </Form.Item>
                    <Form.Item label="Smoking History" name="smokingHistory" rules={[{ required: true, message: 'Please input valid numbers'}]}>
                        <InputNumber min={0} onChange={this.onChangeSmokingHistory} precision={2}/>
                    </Form.Item>
                    <Form.Item label="BMI" name="bmi" rules={[{ required: true, message: 'Please input valid numbers'}]}>
                        <InputNumber min={0} onChange={this.onChangeBmi} precision={2}/>
                    </Form.Item>
                    <Form.Item label="HbA1c Level" name="hbA1cLevel" rules={[{ required: true, message: 'Please input valid numbers'}]}>
                        <InputNumber min={0} onChange={this.onChangeHbA1cLevel} precision={2}/>
                    </Form.Item>
                    <Form.Item label="Blood Glucose Level" name="BloodGlucoseLevel" rules={[{ required: true, message: 'Please input valid numbers'}]}>
                        <InputNumber min={0} onChange={this.onChangeBloodGlucoseLevel} precision={2}/>
                    </Form.Item>
                    
                </Form> 

                <Button  type="primary" htmlType="submit" onClick={this.onClickPrediction}> Get Prediction </Button>



                <div>
                    <Card title="Prediction: " bordered={false} style={{ width: 800, margin: 20}}>
                        <p >
                            {this.state.currentPrediction}
                        </p>
                    </Card>

                </div>

                IntelliHealthBot
                <p style={{fontWeight: "normal", fontSize: "15px"}}>
                    If you have any question, feel free to ask any questions about the diabetes
                </p>


                <TextArea showCount maxLength={100} onChange={this.onChangeQuestions} style={{ width: 800, height: 300 }} />
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



            </div>
        );
    }
}
export default Physician;