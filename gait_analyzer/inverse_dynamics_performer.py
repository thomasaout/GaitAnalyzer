import numpy as np
import biorbd

from gait_analyzer.experimental_data import ExperimentalData


class InverseDynamicsPerformer:
    """
    This class performs the inverse dynamics based on the kinematics and the external forces.
    """

    def __init__(self, 
                 experimental_data: ExperimentalData, 
                 biorbd_model: biorbd.Model, 
                 q: np.ndarray, 
                 qdot: np.ndarray, 
                 qddot: np.ndarray):
        """
        Initialize the InverseDynamicsPerformer.
        .
        Parameters
        ----------
        experimental_data: ExperimentalData
            The experimental data from the trial
        biorbd_model: biorbd.Model
            The biorbd model to use for the inverse dynamics
        q: np.ndarray() # TODO: Charbie -> fill
        """
        
        # Checks
        if not isinstance(experimental_data, ExperimentalData):
            raise ValueError(
                "experimental_data must be an instance of ExperimentalData. You can declare it by running ExperimentalData(file_path)."
            )
        if not isinstance(biorbd_model, biorbd.Model):
            raise ValueError(
                "biorbd_model must be an instance of biorbd.Model. You can declare it by running biorbd.Model('path_to_model.bioMod')."
            )
        
        # Initial attributes
        self.experimental_data = experimental_data
        self.biorbd_model = biorbd_model
        self.q = q
        self.qdot = qdot
        self.qddot = qddot

        # Extended attributes
        self.tau = np.ndarray(())

        # Perform the inverse dynamics
        self.perform_inverse_dynamics()
        
        
    def get_f_ext_for_inverse_dynamics(self):

        contact_names = ["LFoot", "RFoot"]
        f_ext_filtered = np.zeros([2, 6, len(self.q[0, :])])
        moment_origin = np.zeros([2, 3, len(self.q[0, :])])
        center_of_pressure_filtered = np.zeros([2, 3, 20 * len(self.q[0, :])])
        originPf = np.zeros([len(contact_names), 3, 1])
        for contact in range(len(contact_names)):
            f_ext = self.c3d['data']['platform'][contact]['force']
            f_ext_filtered[contact, :, :] = self.forcedatafilter(f_ext, 4, 2000, 20)
            moment_ext = self.c3d['data']['platform'][contact]['moment'] / 1000
            moment_extfilt[contact, :, :] = self.forcedatafilter(moment_ext, 4, 2000, 20)
            cop_ext = self.c3d['data']['platform'][contact]['center_of_pressure'] / 1000
            center_of_pressure_filtered[contact, :, :] = self.forcedatafilter(cop_ext, 4, 2000, 10)
            originPf[contact, 0, 0] = np.mean(self.c3d['data']['platform'][contact]['corners'][0])/1000
            originPf[contact, 1, 0] = np.mean(self.c3d['data']['platform'][contact]['corners'][1])/1000
            originPf[contact, 2, 0] = np.mean(self.c3d['data']['platform'][contact]['corners'][2])/1000

            for ii in range(len(moment_extfilt[contact, 0, :])):
                r = originPf[contact, :, 0]-center_of_pressure_filtered[contact, :, ii]
                M_offset = np.cross(r, f_ext_filtered[contact, :, ii])
                moment_origin[contact, :, ii] = moment_extfilt[contact, :, ii] + M_offset

        moment_extfilt = moment_origin
        #moment_extfilt = np.zeros([len(contact_names), 3, 20 * len(self.q[0, :])])

        self.force = np.empty([2, 9, len(self.q[0, :])])
        PointApplication = np.zeros([2, 3, len(self.q[0, :])])
        
        for i in range(len(self.q[0, :])):
            self.ext_load = self.model.externalForceSet()

            for contact in range(len(contact_names)):
                name = biorbd.String(contact_names[contact])
                spatial_vector = np.concatenate((moment_extfilt[contact, :, 20 * i], f_ext_filtered[contact, :, 20 * i]))
                PointApplication[contact, :, i] = center_of_pressure_filtered[contact, :, 20 * i] #self.c3d['data']['platform'][contact]['origin']
                PA = PointApplication[contact, :, i]
                if spatial_vector[5] > 5:
                    if (PointApplication[contact, 2, i-1] or abs(PA[2]-PointApplication[contact, 2, i-1]) < 0.00001):
                        self.ext_load.add(name, spatial_vector, PA)
                        self.force[contact, 0:3, i] = PA
                        self.force[contact, 3:6, i] = f_ext_filtered[contact, :, 20 * i]
                        self.force[contact, 6:, i] = moment_extfilt[contact, :, 20 * i]
            
        return



    def perform_inverse_dynamics(self):
        """
        Code adapted from ophlariviere's biomechanics_tools
        Modifications:
        - Do not filter q, qdot, qddot since the kalman filter should already do the job.
        - Use only the force plate data at the frames where the marker positions were recorded (taking the mean of frames [i-5:i+5])
        """
        
        tau_data = []
        
        # Smoothing position data
        self.q = savgol_filter(self.q, 30, 3)

        # Initialize arrays for angular velocity and acceleration
        angular_velocity = np.empty_like(self.q)
        angular_acc = np.empty_like(self.q)

        # Calculate angular velocity by taking the gradient of the smoothed position data
        for dof in range(len(self.q[:, 0])):
            angular_velocity[dof, :] = np.gradient(self.q[dof, :], 1 / 100)

        # Apply filtering to the angular velocity data
        self.qdot = self.forcedatafilter(angular_velocity, 4, 100, 10)

        # Calculate angular acceleration by taking the gradient of the filtered velocity data
        for dof in range(len(self.q[:, 0])):
            angular_acc[dof, :] = np.gradient(self.qdot[dof, :], 1 / 100)

        # Apply filtering to the angular acceleration data
        self.qddot = self.forcedatafilter(angular_acc, 4, 100, 10)

        tau = self.model.InverseDynamics(self.q[:, i], self.qdot[:, i], self.qddot[:, i], self.ext_load)
        tau_data.append(tau.to_array())
        
        tau_data = np.array(tau_data)
        self.tau = np.transpose(tau_data)
        #self.tau = self.forcedatafilter(angular_acc, 1, 100, 15)
        self.is_inverse_dynamic_performed = True
        return self.tau


    def inputs(self):
        return {
            "biorbd_model": self.biorbd_model,
            "f_ext": self.f_ext,
            "q": self.q,
            "qdot": self.qdot,
            "qddot": self.qddot,
        }

    def outputs(self):
        return {
            "tau": self.tau,
        }
