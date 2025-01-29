import numpy as np
import biorbd

from gait_analyzer.experimental_data import ExperimentalData


class KinematicsReconstructor:
    """
    This class reconstruct the kinematics based on the marker position and the model predefined.
    """

    def __init__(self, experimental_data: ExperimentalData, biorbd_model, animate_kinematics_flag: bool = False):
        """
        Initialize the Events.
        .
        Parameters
        ----------
        experimental_data: ExperimentalData
            The experimental data from the trial
        """
        # Checks
        if not isinstance(experimental_data, ExperimentalData):
            raise ValueError(
                "experimental_data must be an instance of ExperimentalData. You can declare it by running ExperimentalData(file_path)."
            )

        # Initial attributes
        self.experimental_data = experimental_data
        self.biorbd_model = biorbd_model

        # Extended attributes
        self.q = np.ndarray(())
        self.qdot = np.ndarray(())
        self.qddot = np.ndarray(())
        self.tau = np.ndarray(())

        # Perform the kinematics reconstruction
        self.perform_kinematics_reconstruction()

        if animate_kinematics_flag:
            self.animate_kinematics()

    def perform_kinematics_reconstruction(self):
        self.t, self.q, self.qdot, self.qddot = biorbd.extended_kalman_filter(
            self.biorbd_model, self.experimental_data.c3d_full_file_path
        )

    def inverse_dynamics(self):
        # TODO: Guys -> How can you do an inverse kinematics if you do not know the foot pressure repartition ?
        """
        Original code from Ophe's biomechanics_tools:
        ---------
        ContactName = ["LFoot", "RFoot"]
        tau_data = []
        f_extfilt = np.zeros([len(ContactName), 3, 20 * len(self.q[0, :])])
        moment_extfilt = np.zeros([len(ContactName), 3, 20 * len(self.q[0, :])])
        moment_origin = np.zeros([len(ContactName), 3, 20 * len(self.q[0, :])])
        cop_extfilt = np.zeros([len(ContactName), 3, 20 * len(self.q[0, :])])
        originPf = np.zeros([len(ContactName), 3, 1])
        for contact in range(len(ContactName)):
            f_ext = self.c3d['data']['platform'][contact]['force']
            f_extfilt[contact, :, :] = self.forcedatafilter(f_ext, 4, 2000, 20)
            moment_ext = self.c3d['data']['platform'][contact]['moment'] / 1000
            moment_extfilt[contact, :, :] = self.forcedatafilter(moment_ext, 4, 2000, 20)
            cop_ext = self.c3d['data']['platform'][contact]['center_of_pressure'] / 1000
            cop_extfilt[contact, :, :] = self.forcedatafilter(cop_ext, 4, 2000, 10)
            originPf[contact, 0, 0] = np.mean(self.c3d['data']['platform'][contact]['corners'][0])/1000
            originPf[contact, 1, 0] = np.mean(self.c3d['data']['platform'][contact]['corners'][1])/1000
            originPf[contact, 2, 0] = np.mean(self.c3d['data']['platform'][contact]['corners'][2])/1000

            for ii in range(len(moment_extfilt[contact, 0, :])):
                r = originPf[contact, :, 0]-cop_extfilt[contact, :, ii]
                M_offset = np.cross(r, f_extfilt[contact, :, ii])
                moment_origin[contact, :, ii] = moment_extfilt[contact, :, ii] + M_offset

        moment_extfilt = moment_origin
        #moment_extfilt = np.zeros([len(ContactName), 3, 20 * len(self.q[0, :])])

        self.force = np.empty([2, 9, len(self.q[0, :])])
        PointApplication = np.zeros([2, 3, len(self.q[0, :])])

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


        for i in range(len(self.q[0, :])):
            self.ext_load = self.model.externalForceSet()

            for contact in range(len(ContactName)):
                name = biorbd.String(ContactName[contact])
                spatial_vector = np.concatenate((moment_extfilt[contact, :, 20 * i], f_extfilt[contact, :, 20 * i]))
                PointApplication[contact, :, i] = cop_extfilt[contact, :, 20 * i] #self.c3d['data']['platform'][contact]['origin']
                PA = PointApplication[contact, :, i]
                if spatial_vector[5] > 5:
                    if (PointApplication[contact, 2, i-1] or abs(PA[2]-PointApplication[contact, 2, i-1]) < 0.00001):
                        self.ext_load.add(name, spatial_vector, PA)
                        self.force[contact, 0:3, i] = PA
                        self.force[contact, 3:6, i] = f_extfilt[contact, :, 20 * i]
                        self.force[contact, 6:, i] = moment_extfilt[contact, :, 20 * i]

            tau = self.model.InverseDynamics(self.q[:, i], self.qdot[:, i], self.qddot[:, i], self.ext_load)
            tau_data.append(tau.to_array())

        tau_data = np.array(tau_data)
        self.tau = np.transpose(tau_data)
        #self.tau = self.forcedatafilter(angular_acc, 1, 100, 15)
        self.is_inverse_dynamic_performed = True
        return self.tau
        """
        pass

    def animate_kinematics(self):
        """
        Animate the kinematics
        """
        # TODO: Charbie -> Animate the kinematics using pyorerun
        pass

    def inputs(self):
        return {
            "biorbd_model": self.biorbd_model,
            "c3d_full_file_path": self.experimental_data.c3d_full_file_path,
        }

    def outputs(self):
        return {
            "q": self.q,
            "qdot": self.qdot,
            "qddot": self.qddot,
            "tau": self.tau,
        }
