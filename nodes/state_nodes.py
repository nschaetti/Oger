#!/usr/bin/env python
# -*- coding: utf-8 -*-

import mdp
import numpy as np


#
# CLASS JoinedStatesNode
# A node doing the union of several state of the reservoir to do the linear regression on it
#
class JoinedStatesNode(mdp.Node):
    """
    A node doing the union of several states of the reservoir to do the linear regression on it
    """

    # Constructor
    def __init__(self, input_dim=100, joined_size=1, fill_before=False, dtype='float64'):
        super(JoinedStatesNode, self).__init__(input_dim=input_dim, dtype=dtype)

        # Variables
        self._joined_size = int(joined_size)
        self._reservoir_size = input_dim
        self._fill_before = fill_before
    # end __init__

    # This node is not trainable
    def is_trainable(self):
        """
        This node is not trainable
        :return:
        """
        return False
    # end is_trainable

    # Execute this node
    def _execute(self, x):
        """
        Execute this node.
        :param x:
        :return:
        """
        # If not overlap, just reshape
        if self._joined_size == 1:
            if not self._fill_before:
                return x
            else:
                return np.vstack((np.zeros(self._reservoir_size), x))
            # end if
        else:
            # Create empty space for joined states
            if self._fill_before:
                n_before = 1
            else:
                n_before = 0
            # end if
            states = np.zeros((x.shape[0] + n_before, self._joined_size * self._reservoir_size))

            # Go through all initial states
            for i in np.arange(0, x.shape[0]):
                # Position
                start_pos = i - self._joined_size + 1
                end_pos = i

                # Under zero
                if start_pos < 0:
                    state = np.zeros(self._joined_size * self._reservoir_size)
                    size = (end_pos+1) * self._reservoir_size
                    state[-size:] = x[0:end_pos+1, :].flatten()
                else:
                    state = x[start_pos:end_pos+1, :].flatten()
                # end if

                # Add to new states
                states[i+n_before, :] = state
            # end for
            return states
        # end if
    # end _execute

# end JoinedStatesNode

#
# CLASS LastTwoStateNode
# Un noeud qui renvoi seulement le dernier états du réservoir pour chaque images traitées
#
class LastTwoStateNode(mdp.Node):
    ##############################################################
    # Constructeur
    ##############################################################
    def __init__(self, mnist_space=0, image_size=28, input_dim=100, dtype='float64'):
        super(LastTwoStateNode, self).__init__(input_dim=input_dim, dtype=dtype)

        # Variables
        self.imagesSize = image_size
        self.interImagesSpace = mnist_space
        self.entrySize = self.imagesSize + self.interImagesSpace
        self.reservoirSize = input_dim

    ##############################################################
    # On entraîne pas ce noeud
    ##############################################################
    def is_trainable(self):
        return False

    ##############################################################
    # Exécution du noeud
    ##############################################################
    def _execute(self, x):
        last1 = x[np.arange(self.entrySize - 2, x.shape[0] + 1, self.entrySize), :]
        last2 = x[np.arange(self.entrySize - 1, x.shape[0] + 1, self.entrySize), :]

        # Retourne le dernier état de chaque images
        return np.hstack((last1, last2))


# A node which returns only the last state of the reservoir
class LastStateNode(mdp.Node):
    """
    A node which returns only the last state of the reservoir
    """

    # Constructor
    def __init__(self, input_dim=100, dtype='float64'):
        """
        Constructor
        :param input_dim:
        :param dtype:
        """
        super(LastStateNode, self).__init__(input_dim=input_dim, dtype=dtype)
    # end __init__

    # Cannot train this node
    def is_trainable(self):
        """
        Cannot train this node
        :return:
        """
        return False
    # end is_trainable

    # Execute this node
    def _execute(self, x):
        """
        Execute this node
        :param x:
        :return:
        """
        # Retourne le dernier état de chaque images
        return x[-1, :]
    # end _execute

# end LastStateNode


#
# CLASS MixedStateNode
# Un noeud qui renvoi l'état du milieu et le dernier état mixés
#
class MixedStateNode(mdp.Node):
    ##############################################################
    # Constructeur
    ##############################################################
    def __init__(self, mnist_space=0, image_size=28, input_dim=100, dtype='float64'):
        super(MixedStateNode, self).__init__(input_dim=input_dim, dtype=dtype)

        # Variables
        self.imagesSize = image_size
        self.interImagesSpace = mnist_space
        self.entrySize = self.imagesSize + self.interImagesSpace
        self.reservoirSize = input_dim

    ##############################################################
    # On entraîne pas ce noeud
    ##############################################################
    def is_trainable(self):
        return False

    ##############################################################
    # Exécution du noeud
    ##############################################################
    def _execute(self, x):
        middle = x[np.arange(int(self.entrySize / 2.0) - 1, x.shape[0], self.entrySize), :]
        last = x[np.arange(self.entrySize - 1, x.shape[0] + 1, self.entrySize), :]

        # Retourne le dernier état de chaque images
        return np.hstack((middle, last))


#
# CLASS MixedThreeStateNode
# Un noeud qui renvoi l'état du milieu et le dernier état mixés
#
class MixedThreeStateNode(mdp.Node):
    ##############################################################
    # Constructeur
    ##############################################################
    def __init__(self, mnist_space=0, image_size=28, input_dim=100, dtype='float64'):
        super(MixedThreeStateNode, self).__init__(input_dim=input_dim, dtype=dtype)

        # Variables
        self.imagesSize = image_size
        self.interImagesSpace = mnist_space
        self.entrySize = self.imagesSize + self.interImagesSpace
        self.reservoirSize = input_dim

    ##############################################################
    # On entraîne pas ce noeud
    ##############################################################
    def is_trainable(self):
        return False

    ##############################################################
    # Exécution du noeud
    ##############################################################
    def _execute(self, x):
        first = x[np.arange(int(self.entrySize / 3.0) - 1, x.shape[0], self.entrySize), :]
        second = x[np.arange(int(self.entrySize / 3.0) * 2 - 1, x.shape[0], self.entrySize), :]
        last = x[np.arange(self.entrySize - 1, x.shape[0] + 1, self.entrySize), :]

        # Retourne le dernier état de chaque images
        return np.hstack((first, second, last))


#
# CLASS MixedThreeStateNode
# Un noeud qui renvoi l'état du milieu et le dernier état mixés
#
class MixedFourStateNode(mdp.Node):
    ##############################################################
    # Constructeur
    ##############################################################
    def __init__(self, mnist_space=0, image_size=28, input_dim=100, dtype='float64'):
        super(MixedFourStateNode, self).__init__(input_dim=input_dim, dtype=dtype)

        # Variables
        self.imagesSize = image_size
        self.interImagesSpace = mnist_space
        self.entrySize = self.imagesSize + self.interImagesSpace
        self.reservoirSize = input_dim

    ##############################################################
    # On entraîne pas ce noeud
    ##############################################################
    def is_trainable(self):
        return False

    ##############################################################
    # Exécution du noeud
    ##############################################################
    def _execute(self, x):
        first = x[np.arange(int(self.entrySize / 4.0) + 1, x.shape[0], self.entrySize), :]
        second = x[np.arange((int(self.entrySize / 4.0) + 1) * 2, x.shape[0], self.entrySize), :]
        third = x[np.arange((int(self.entrySize / 4.0) + 1) * 3, x.shape[0], self.entrySize), :]
        last = x[np.arange(self.entrySize - 1, x.shape[0] + 1, self.entrySize), :]

        # Retourne le dernier état de chaque images
        return np.hstack((first, second, third, last))


#
# CLASS TrajectoryStatesNode
# Un noeud qui fait l'union de plusieurs états du réservoir et en déduit la trajéctoire de l'espace d'états
#
class TrajectoryStatesNode(mdp.Node):
    ##############################################################
    # Constructeur
    ##############################################################
    def __init__(self, mnist_space=0, image_size=28, input_dim=100, dtype='float64'):
        super(TrajectoryStatesNode, self).__init__(input_dim=input_dim, dtype=dtype)

        # Variables
        self.imagesSize = image_size
        self.interImagesSpace = mnist_space
        self.entrySize = self.imagesSize + self.interImagesSpace
        self.reservoirSize = input_dim

    ##############################################################
    # On entraîne pas ce noeud
    ##############################################################
    def is_trainable(self):
        return False

    ##############################################################
    # Exécution du noeud
    ##############################################################
    def _execute(self, x):

        # Nombre d'éléments à la sortie
        nbOut = int(x.shape[0] / self.entrySize)

        # Pour chaque entrée
        for i in np.arange(0, x.shape[0]):
            if i < x.shape[0] - 1:
                x[i, :] = x[i + 1, :] - x[i, :]
            else:
                x[i, :] = np.zeros(x.shape[1])

        # Reforme
        x.shape = (nbOut, self.reservoirSize * self.entrySize)

        return x


#
# CLASS DigitSingleClassifierNode
# Un noeud qui permet de récupérer le digit élu par une seule sortie du réservoir
#
class DigitSingleClassifierNode(mdp.Node):
    ##############################################################
    # Constructeur
    ##############################################################
    def __init__(self, mnist_space=0, label_space_ratio=0, digit_space_ratio=0, image_size=28, nb_digit=10,
                 method='average', input_dim=10, analyze_sampling=20, dtype='float64'):
        super(DigitClassifierNode, self).__init__(input_dim=input_dim, dtype=dtype)

        # Variables globales
        self.interImagesSpace = mnist_space
        self.interImagesRatio = label_space_ratio
        self.digitImageRatio = digit_space_ratio
        self.imagesSize = image_size
        self.nbDigits = nb_digit
        self.entrySize = self.imagesSize + self.interImagesSpace
        self.Method = method
        self.analyzeSampling = analyze_sampling

    ##############################################################
    # On ne peut pas entraîner ce noeud
    ##############################################################
    def is_trainable(self):
        return False

    ##############################################################
    # Exécution du noeud
    ##############################################################
    def _execute(self, x):

        # Sortie
        y = np.array([])

        # Vérifie les données
        if x.shape[1] != 1:
            raise InvalidDigitNumber("Nombre de digits en entrée invalide")
        if x.shape[0] % self.entrySize != 0:
            raise InvalidTimeserie(
                "Séries temporelles en entrées invalide, taille multiple de {} attendue".format(self.entrySize))

        # Pour chaque partie des séries correspondante à un digit
        for i in np.arange(0, x.shape[0], self.entrySize):
            # Le début et la fin de la partie à examiner
            start = i + int(self.imagesSize * self.digitImageRatio)
            end = i + self.entrySize + int(self.interImagesRatio * self.interImagesSpace)

            # Résultat
            average = int(round(np.average(np.flatten(x[start:end]))))

            # Inscrit le vainqueur
            y = np.append(y, average)

        # Renvoi le résultat
        return (y, x[0:self.analyzeSampling * self.entrySize, :])


#
# CLASS SingleInputClassifier
# Un noeud qui classifie plusieurs sortie par digit pour un seul digit
#
class SingleInputClassifier(mdp.Node):
    ##############################################################
    # Constructeur
    ##############################################################
    def __init__(self, input_dim=10, output_dim=1, dtype='float64'):
        super(SingleInputClassifier, self).__init__(input_dim=input_dim, dtype=dtype)

    ##############################################################
    # On ne peut pas entraîner ce noeud
    ##############################################################
    def is_trainable(self):
        return False

    ##############################################################
    # Exécution du noeud
    ##############################################################
    def _execute(self, x):
        # Sortie
        y = np.array([])

        # Pour chaque élément, on cherche le plus grosse sortie
        for i in np.arange(x.shape[0]):
            omax = np.argmax(x[i, :])
            y = np.append(y, omax)

        return y


#
# CLASS DigitClassifierNode
# Un noeud qui permet de récupérer le digit élu par les sorties du réservoir
#
class DigitClassifierNode(mdp.Node):
    ##############################################################
    # Constructeur
    ##############################################################
    def __init__(self, mnist_space=0, label_space_ratio=0, digit_space_ratio=0, image_size=28, nb_digit=10,
                 method='average', input_dim=10, analyze_sampling=20, dtype='float64'):
        super(DigitClassifierNode, self).__init__(input_dim=input_dim, dtype=dtype)

        # Variables globales
        self.interImagesSpace = mnist_space
        self.interImagesRatio = label_space_ratio
        self.digitImageRatio = digit_space_ratio
        self.imagesSize = image_size
        self.nbDigits = nb_digit
        self.entrySize = self.imagesSize + self.interImagesSpace
        self.Method = method
        self.analyzeSampling = analyze_sampling

    ##############################################################
    # On ne peut pas entraîner ce noeud
    ##############################################################
    def is_trainable(self):
        return False

    ##############################################################
    # On cherche la série qui a un point maximum
    ##############################################################
    def searchMaximum(self, x):
        # Cherche le maximum
        maximum = 0
        max_digit = -1
        for j in np.arange(0, x.shape[0]):
            if np.amax(x[j, :]) > maximum:
                maximum = np.amax(x[j, :])
                max_digit = np.argmax(x[j, :])
        return maximum, max_digit

    ##############################################################
    # On cherche la série qui a la plus haut moyenne
    ##############################################################
    def searchAverage(self, x):
        # Calcule la moyenne
        counters = np.zeros(self.nbDigits, dtype='float64')
        for j in np.arange(0, x.shape[0]):
            counters += x[j, :]
        counters /= float(x.shape[0])
        return np.amax(counters), np.argmax(counters)

    ##############################################################
    # On cherche la série qui a le plus haut niveau de fin
    ##############################################################
    def searchLast(self, x):
        maximum = np.amax(x[-1, :])
        max_digit = np.argmax(x[-1, :])
        return maximum, max_digit

    ##############################################################
    # Exécution du noeud
    ##############################################################
    def _execute(self, x):
        # Sortie
        y = np.array([])

        # Vérifie les données
        if x.shape[1] != self.nbDigits:
            raise InvalidDigitNumber("Nombre de digits en entrée invalide")
        if x.shape[0] % self.entrySize != 0:
            raise InvalidTimeserie(
                "Séries temporelles en entrées invalide, taille multiple de {} attendue".format(self.entrySize))

        # Pour chaque partie des séries correspondante à un digit
        for i in np.arange(0, x.shape[0], self.entrySize):
            # Le début et la fin de la partie à examiner
            start = i + int(self.imagesSize * self.digitImageRatio)
            end = i + self.entrySize + int(self.interImagesRatio * self.interImagesSpace)

            # Cherche le maximum
            if self.Method == 'average':
                maximum, max_digit = self.searchAverage(x[start:end, :])
            elif self.Method == 'max':
                maximum, max_digit = self.searchMaximum(x[start:end, :])
            elif self.Method == 'last':
                maximum, max_digit = self.searchLast(x[start:end, :])

            # Inscrit le vainqueur
            y = np.append(y, max_digit)

        # Renvoi le résultat
        return (y, x[0:self.analyzeSampling * self.entrySize, :])


#
# CLASS InvalidDigitNumber
# Nombre de digits invalide
#
class InvalidDigitNumber(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


#
# CLASS InvalidTimeserie
# Séries temporelles invalides
#
class InvalidTimeserie(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)