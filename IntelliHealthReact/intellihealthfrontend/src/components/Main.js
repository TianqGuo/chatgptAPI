import React from 'react';
// import '../styles/App.css';
import Welcome from './Welcome';
// import Patient from './Patient';
import Physician from './Physician';


// destructuring

// define component app class
class App extends React.Component {
    constructor(props) {
        super(props);
        // define state for the components states
        this.state = {
            phase: 0,
        }
    }

    phases = {
        welcome : 0,
        patient : 1,
        physician : 2

    }

    setStatePhase = async (curPhase) => {
        console.log("current phase: " + this.state.phase);
        await this.setState({ phase: curPhase });
        console.log("current phase: " + this.state.phase);
    }

    render = () => (
        <div className="App">
            {
                this.state.phase === this.phases.welcome && 
                <Welcome setStatePhase = {this.setStatePhase} phases = {this.phases} /> 
            }

            {/* {
                this.state.phase === this.phases.patient && 
                <Patient setStatePhase = {this.setStatePhase} phases = {this.phases} />
            } */}

            {
                this.state.phase === this.phases.physician && 
                <Physician setStatePhase = {this.setStatePhase} phases = {this.phases} />
            }

        </div>
    )
}
 
export default App;
