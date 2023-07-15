import React, {Component} from 'react';
import { Button } from 'antd';

class Welcome extends Component {
    onClickPatient = () => {
        this.props.setStatePhase(this.props.phases.patient);
    }

    onClickPhysician = () => {
        this.props.setStatePhase(this.props.phases.physician);
    }

    render() {
        return (
            <div className="Main" style={{fontWeight: "bold"}}>
            Welcome to IntelliHealth 
            <p>
            </p>
            <Button type="primary" htmlType="submit" onClick = {this.onClickPatient}> I am a patient</Button>
            <p>
            </p>
            <Button type="primary" htmlType="submit" onClick = {this.onClickPhysician}> I am a physician </Button>
        </div>
        );
    }
}
export default Welcome;